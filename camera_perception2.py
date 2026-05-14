import numpy as np
import robotic as ry
from typing import Tuple, List, Any
import cv2 as cv
import pyrealsense2 as rs
import matplotlib.pyplot as plt



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
    name: str = "l_cameraWrist",
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
        rgb, depth, pcl = bot.getImageDepthPcl(name)
        fxycxy = bot.getCameraFxycxy(name)
        """
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
        """
        return rgb, depth, pcl,fxycxy
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

def segment_dice_by_depth(depth, z_min, z_max):
    depth_m = depth.astype(np.float32)
    if depth_m.max() > 100: # assume depth is in mm in realsense
        depth_m = depth_m / 1000.0
    mask = (depth_m > z_min) & (depth_m < z_max)
    mask = mask.astype(np.uint8) * 255
    
    mask = cv.medianBlur(mask, 5) # avoid glare on the dice
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

def split_left_right(roi):
       h,w = roi.shape[:2]
       left = roi[:, :w//2]
       right = roi[:, w//2:]
       return left, right

def observe_and_detect_dice(C, bot, q_home, args):
    # move to camera observation pose
    ret_cam, q_cam = camera_ik(C, "camera_obs", q_home)
    if not ret_cam.feasible:
        print("Camera IK not feasible")
        return None

    bot.moveTo(q_cam[0])
    bot.wait(C, forKeyPressed=True, forTimeToEnd=True)

    # capture images
    if args.real:
        rgb, depth, points, fxycxy = get_wrist_camera(
            source="real", bot=bot, name="l_cameraWrist", serial="None"
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

    # dice detection
    z_min = round(depth.min() , 2)
    z_max = round(depth.min() * 1.05, 2)
    if z_max <= z_min:
        z_max = z_min + 0.02
    
    mask = segment_dice_by_depth(depth, z_min=z_min, z_max=z_max)
    cv.imshow("Mask_top", mask)
    roi = apply_mask(rgb, mask)
    cv.imshow("ROI_top", roi)
    pip_count, circles = detect_pip_hough(roi)
    print("Total Dice number: ", pip_count)
    print("Pips Positions: ", circles)
    
    # side face dice detection
    mask = segment_dice_by_depth(depth, z_min=z_max, z_max=depth.max())
    cv.imshow("Mask_side", mask)
    roi = apply_mask(rgb, mask)
    # Split ROI into left/right dice
    roi_left, roi_right = split_left_right(roi)
    cv.imshow("ROI Left", roi_left)
    cv.imshow("ROI Right", roi_right)
    pip_count_left, circles_left = detect_pip_hough(roi_left)
    pip_count_right, circles_right = detect_pip_hough(roi_right)
    print("Left dice pip count:", pip_count_left)
    print("Right dice pip count:", pip_count_right)

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