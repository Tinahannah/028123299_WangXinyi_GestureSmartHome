from ultralytics import YOLO
import cv2

# Load a pretrained YOLO model for pose estimation
model = YOLO("./total.pt")  # Make sure the model file is in the correct path

# Open the camera (camera 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Perform pose estimation on the frame
    # results = model(frame)
    results = model(frame)
    print(results[0].speed)  # 显示预处理/推理/后处理时间(ms)
    print(results[0].boxes.conf)  # 显示检测置信度

    # Visualize the results on the frame
    annotated_frame = results[0].plot()

    # Display the annotated frame
    cv2.imshow("YOLO Pose Estimation", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()