import numpy as np
import robotic as ry
from typing import Tuple, List, Any
import cv2 as cv
import pyrealsense2 as rs
import matplotlib.pyplot as plt


def add_wrist_camera(
    C: ry.Config
):
    """
    add a wrist camera frame (simulation use) on the robot gripper.
    """
    f = C.getFrame("cameraWrist")
    f.setShape(type=ry.ST.marker, size=[0.08])
    f.setColor([1.0, 1.0, 1.0])
    C.view()
    return f

### Realsense Utils ###

def start_realsense(serial=None, width=1280, height=720, fps=30):
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(config)
    # Set up alignment so depth matches color frame
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, profile
    
def get_color_intrinsics(profile):
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intr = color_profile.get_intrinsics()
    # intr: width, height, ppx, ppy, fx, fy, model, coeffs (5)
    K = np.array([[intr.fx, 0, intr.ppx],
                  [0, intr.fy, intr.ppy],
                  [0, 0, 1]], dtype=np.float64)
    # Map RS distortion to OpenCV k1,k2,p1,p2,k3 (Brown-Conrady)
    if intr.model in (rs.distortion.brown_conrady, rs.distortion.inverse_brown_conrady):
        coeffs = np.array(intr.coeffs[:5], dtype=np.float64)
    else:
        coeffs = np.zeros(5, dtype=np.float64)
    return K, coeffs, intr.width, intr.height
#—————————————————Realsense Utils————————————————————————————

def get_wrist_camera(
    source: str,
    bot: ry.BotOp = None,
    name: str = "cameraWrist",
    serial: str = None,
):
    """
    source: 'sim' or 'real'
    return: rgb, depth, points_or_None, intrinsics_or_None
    """
    if source == "sim":
        rgb, depth, points = bot.getImageDepthPcl(name)
        fxycxy = bot.getCameraFxycxy(name) #The intrinsics are given by the focal lengths (f_x, f_y) and image center (c_x, c_y)
        return rgb, depth, points, fxycxy

    elif source == "real":
        pipeline, profile = start_realsense(serial=serial)
        try:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            rgb = np.asanyarray(color_frame.get_data())
            depth = np.asanyarray(depth_frame.get_data())
            K, dist, W, H = get_color_intrinsics(profile)
            return rgb, depth, None, (K, dist, W, H)
        finally:
            pipeline.stop()

    else:
        raise ValueError("source must be 'sim' or 'real'")


def camera_ik(C: ry.Config,
    goal_pos: str,
    q_home: np.ndarray,
) -> Tuple[ry.SolverReturn, ry.KOMO, np.ndarray]:
    komo = ry.KOMO(C, 1, 1, 0, False)  # 1 phase, 1 slice, kOrder=0 IK
    komo.addObjective([], ry.FS.jointState, [], ry.OT.sos, [1e-1], q_home)
    komo.addObjective([], ry.FS.accumulatedCollisions, [], ry.OT.eq)
    komo.addObjective([], ry.FS.jointLimits, [], ry.OT.ineq)
    
    komo.addObjective([], ry.FS.positionDiff, ["cameraWrist", goal_pos], ry.OT.eq, [1e1])
    komo.addObjective([], ry.FS.scalarProductZZ, ["cameraWrist", goal_pos], ry.OT.eq, [1e1], [-1])
    komo.addObjective([], ry.FS.scalarProductYY, ["cameraWrist", goal_pos], ry.OT.eq, [1e1], [-1])
    
     # Solve the NLP
    ret = ry.NLP_Solver(komo.nlp(), verbose=0).solve()
    # Check solution feasibility
    q = None
    if ret.feasible:
        print("-- Camera_obs is feasible")
        print(f"ret={ret}")
        q = komo.getPath()
    else:
        print("-- Camera_obs is infeasible!")

    return ret, q

# Detect dice Number in image

def face_mask(depth, z_min, z_max):
    depth_m = depth.astype(np.float32)
    if depth_m.max() > 100: # assume depth is in mm in realsense
        depth_m = depth_m / 1000.0
    mask = (depth_m > z_min) & (depth_m < z_max)
    mask = mask.astype(np.uint8) * 255
    
    mask = cv.medianBlur(mask, 3) # avoid glare on the dice
    return mask

def apply_mask(rgb, mask): #ROI: Region of Interest
    return cv.bitwise_and(rgb, rgb, mask=mask)

def detect_pip_hough(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)
    gray = cv.equalizeHist(gray)

    circles = cv.HoughCircles( #Finds circles in a grayscale image using the Hough transform.
        gray,
        cv.HOUGH_GRADIENT,
        dp=1.,
        minDist=10,
        param1=80,
        param2=10,
        minRadius=3,
        maxRadius=15
    )

    pip_count = 0 if circles is None else circles.shape[1]
    return pip_count, circles

# Side face mask
def split_left_right(roi):
       h,w = roi.shape[:2]
       left = roi[:, :w//2]
       right = roi[:, w//2:]
       return left, right
   
def preprocess_for_blobs(roi):
    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    bin_img = cv.adaptiveThreshold(
        gray, 255,
        cv.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv.THRESH_BINARY_INV,
        21, 3
    )

    kernel = np.ones((3, 3), np.uint8)
    bin_img = cv.morphologyEx(bin_img, cv.MORPH_CLOSE, kernel, iterations=1)
    return bin_img

def detect_pips_blob(roi_bgr):
    bin_img = preprocess_for_blobs(roi_bgr)

    params = cv.SimpleBlobDetector_Params()
    params.filterByColor = False

    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 500

    params.filterByCircularity = True
    params.minCircularity = 0.2

    params.filterByConvexity = True
    params.minConvexity = 0.5

    params.filterByInertia = True
    params.minInertiaRatio = 0.1

    detector = cv.SimpleBlobDetector_create(params)
    keypoints = detector.detect(bin_img)

    return len(keypoints), keypoints, bin_img

def find_largest_quad(gray):
    edges = cv.Canny(gray, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    quad = None
    best_area = 0
    for c in contours:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            a = cv.contourArea(approx)
            if a > best_area:
                best_area = a
                quad = approx
    return quad

def warp_face(roi_bgr, quad, out_size=200):
    pts = quad.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]

    src = np.array([tl, tr, br, bl], dtype=np.float32)
    dst = np.array([[0, 0], [out_size-1, 0],
                    [out_size-1, out_size-1], [0, out_size-1]], dtype=np.float32)

    M = cv.getPerspectiveTransform(src, dst)
    warped = cv.warpPerspective(roi_bgr, M, (out_size, out_size))
    return warped

def detect_side_pips(roi_bgr):
    gray = cv.cvtColor(roi_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    quad = find_largest_quad(gray)
    if quad is None:
        return 0, None, None

    warped = warp_face(roi_bgr, quad, out_size=200)
    count, kps, bin_img = detect_pips_blob(warped)
    return count, warped, bin_img


def observe_and_detect_dice(C, bot, q_home, args):
    # move to camera observation pose
    ret_cam, q_cam = camera_ik(C, "camera_obs", q_home)
    if not ret_cam.feasible:
        print("Camera IK not feasible")
        return None

    bot.moveTo(q_cam[0])
    bot.wait(C, forKeyPressed=True, forTimeToEnd=True)

    # capture images from real or simulation
    if args.real:
        rgb, depth, points, fxycxy = get_wrist_camera(
            source="real", bot=bot, name="cameraWrist", serial="None"
        )
    else:
        rgb, depth, points, fxycxy = get_wrist_camera(
            source="sim", bot=bot, name="cameraWrist", serial="None"
        )

    # display rgb/depth
    cv.imshow("RGB Image", rgb)
    depth_normalized = cv.normalize(depth, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    cv.imshow("Depth Image", depth_normalized)

    print("Depth: ", depth.dtype, depth.min(), depth.max())
    print("Depth Normalized: ", depth_normalized.dtype, depth_normalized.min(), depth_normalized.max())
    table = C.getFrame("table")
    table_pos = table.getPosition()
    print("Table height: ", table_pos[2])

    # top face dice detection
    mask = face_mask(depth, z_min=0.74, z_max=0.75)
    cv.imshow("Front face Mask", mask)
    roi = apply_mask(rgb, mask)
    cv.imshow("Front face ROI", roi)
    pip_count, circles = detect_pip_hough(roi)
    print("Total Dice number: ", pip_count)
    print("Pips Positions: ", circles)

    # side face dice detection
    mask = face_mask(depth, z_min=0.745, z_max=0.79)
    cv.imshow("Side face Mask", mask)
    roi = apply_mask(rgb, mask)
    cv.imshow("Side face ROI", roi)
    
    # Split ROI into left/right dice
    roi_left, roi_right = split_left_right(roi)
    # Detect pips for left dice
    count_left, warped_left, bin_left = detect_side_pips(roi_left)
    if warped_left is not None:
        cv.imshow("Left Face Warped", warped_left)
        cv.imshow("Left Face Bin", bin_left)
    print("Left dice pip count:", count_left)

    # Detect pips for right dice
    count_right, warped_right, bin_right = detect_side_pips(roi_right)
    if warped_right is not None:
        cv.imshow("Right Face Warped", warped_right)
        cv.imshow("Right Face Bin", bin_right)
    print("Right dice pip count:", count_right)
    
    cv.waitKey(0)
    cv.destroyAllWindows()

    # show point cloud
    if points is not None:
        pclFrame = C.addFrame('pcl', 'cameraWrist')
        pclFrame.setPointCloud(points, rgb)
        pclFrame.setColor([1., 0., 1.])
        C.view()
        C.delFrame('pcl')

    return pip_count