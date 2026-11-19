import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. 配置识别器选项
model_path = 'gesture_recognizer.task' # 这个文件在代码首次运行时会被自动下载
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# 2. 打开摄像头
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("摄像头未能正常读取，请检查设备。")
        break

    # 3. 将 OpenCV 的 BGR 图像转为 MediaPipe 要求的 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 4. 进行手势识别
    recognition_result = recognizer.recognize(mp_image)

    # 5. 处理并显示识别结果
    if recognition_result.gestures:
        # 获取识别到的第一个手势
        top_gesture = recognition_result.gestures[0][0]
        gesture_name = top_gesture.category_name
        score = top_gesture.score

        # 如果识别到 "Open_Palm"，就在画面上显示 "OPEN"！
        if gesture_name == "Open_Palm":
            cv2.putText(frame, "OPEN", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
        else:
            # 也可以显示识别到的其他手势名称，方便调试
            cv2.putText(frame, f'{gesture_name} ({score:.2f})', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "No Gesture", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 6. 显示处理后的画面
    cv2.imshow('Hand Gesture Recognition', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()