import cv2
import numpy as np
import pupil_apriltags as apriltag
import torch
from pupil_apriltags import Detector
import math

# --- 1. Define Camera Parameters ---
# These values are examples. Replace with your calibrated camera's values.
fx = 966.86304605  # Focal length x
fy = 966.10858656  # Focal length y
cx = 651.78364438  # Principal point x
cy = 400.56840661  # Principal point y
mtx = np.array([[1133.86304605, 0, 651.7836443], [0, 1133.10858656, 400.56840661], [0, 0, 1]])
# dist = np.array([ .000151605783, .000-707863956,  .000412906296, -.000691293906, 1.22389446])# Assuming zero distortion for simplicity. Use your calibrated values.
dist = np.array([0,0,0,0,0])

# --- 2. Define Tag Parameters ---
tag_size = 0.16 # in meters; physical size of your printed tag
tag_family = 'tag36h11'

# --- 3. Initialize AprilTag Detector ---
detector = apriltag.Detector(families=tag_family)

# --- 4. Define 3D object points of the tag ---
# These are the corners of the tag in its own coordinate system.
# The corners are in counter-clockwise order, starting from the bottom-left.
half_size = tag_size / 2
tag_points = np.array([
    [-half_size, -half_size, 0],
    [half_size, -half_size, 0],
    [half_size, half_size, 0],
    [-half_size, half_size, 0]
])

# --- 5. Instantiate MDE models ---
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# --- 6. Main Detection Loop ---
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # MiDaS MDE Depth Map

    midas_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(midas_frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=midas_frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
    depth_map_colors = (depth_map*255).astype(np.uint8)
    depth_map_colors = cv2.applyColorMap(depth_map_colors, cv2.COLORMAP_MAGMA)

    cv2.imshow("Depth Map", depth_map_colors)

    # AprilTag Detection

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(gray)
    
    absolute_depth_map = None

    if detections:
        for i, detection in enumerate(detections):
            # Extract 2D image points from the detection
            image_points = detection.corners.reshape(4, 2)

            # --- 1. Estimate the 3D pose using solvePnP ---
            # solvePnP finds the camera's pose relative to the tag's coordinate system.
            success, rvec, tvec = cv2.solvePnP(
                tag_points, 
                image_points, 
                mtx, 
                dist,
                flags=cv2.SOLVEPNP_ITERATIVE
            )

            if success:
                # --- 2. Draw Visualization ---
                # Draw the detected tag outline
                cv2.polylines(frame, [np.int32(image_points)], True, (0, 255, 0), 2)
                
                # Draw the 3D axes on the tag
                cv2.drawFrameAxes(frame, mtx, dist, rvec, tvec, length=tag_size, thickness=3)

                # --- 3. Display Pose Data ---
                # Position (tvec) and Orientation (rvec)
                text = f"ID: {detection.tag_id} | Pos: ({tvec[0][0]:.2f}, {tvec[1][0]:.2f}, {tvec[2][0]:.2f})m"
                cv2.putText(frame, text, (int(image_points[0][0]), int(image_points[0][1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                true_depth = tvec[2][0]
                print(f"AprilTag ID: {detection.tag_id}, True Depth: {true_depth:.2f}m")
                
                

                # -- 4. Calculate relative depth of AprilTag
                y1 = math.floor(image_points[3][1])
                y2 = math.floor(image_points[0][1]+1)
                x1 = math.floor(image_points[3][0])
                x2 = math.floor(image_points[0][0]+1)
                """
                print("coordinates")
                print(y1)
                print(y2)
                print(x1)
                print(x2)
                """
                if y1 < y2 and x1 < x2:
                    april_tag_depth = depth_map[y1:y2,x1:x2]
                    april_tag_depth_val = np.mean(april_tag_depth)
                    print(april_tag_depth_val) # relative depth value from midas around the april taa
                    april_tag_true_depth = 0.0
                    

                    # using the april tag depth, multiply entire relative depth map frame by (true april tag depth / relative april tag depth) to get absolute depth map
                    if april_tag_depth_val > 0:
                        scale = true_depth / april_tag_depth_val
                        absolute_depth_map = depth_map * scale

                        print(f"True depth: {true_depth:.3f} m | MiDaS val: {april_tag_depth_val:.3f} | Scale: {scale:.3f}")
                    
                    
    # Display the result
    cv2.imshow("3D AprilTag Detection", frame)
    
    if absolute_depth_map is not None:
        print(np.mean(absolute_depth_map))
        abs_norm = cv2.normalize(absolute_depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)
        abs_colors = (abs_norm * 255).astype(np.uint8)
        abs_colors = cv2.applyColorMap(abs_colors, cv2.COLORMAP_VIRIDIS)
        cv2.imshow("Absolute Depth Map (meters, scaled)", abs_colors)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()