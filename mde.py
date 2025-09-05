import cv2
import torch
import numpy as np
import time

camera_id = 0

cap = cv2.VideoCapture(camera_id)

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")   
        break
    
    start = time.time()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    input_batch = transform(frame).to(device)

    with torch.no_grad():
        prediction = midas(input_batch)

        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64F)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    depth_map = (depth_map*255).astype(np.uint8)
    depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)

    cv2.putText(frame, f'FPS: {int(fps)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
    cv2.imshow("Depth Map", depth_map)
    cv2.imshow("External Camera Feed", frame)

    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()