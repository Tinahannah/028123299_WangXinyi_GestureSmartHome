import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ===== 1. 加载模型 =====
model_path = 'gesture_recognizer.task'  # 确保这个文件在代码同目录
base_options = python.BaseOptions(
    model_asset_path=model_path,
    delegate=python.BaseOptions.Delegate.CPU  # Mac用户需要这行
)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ===== 2. 定义手势映射（英文名 -> 显示名 + 颜色） =====
gesture_config = {
    "Open_Palm":   {"label": "OPEN",  "color": (0, 255, 0)},   # 绿色
    "Closed_Fist": {"label": "CLOSE", "color": (0, 0, 255)},   # 红色
    "Thumb_Up":    {"label": "UP",    "color": (255, 0, 0)},   # 蓝色
    "Thumb_Down":  {"label": "DOWN",  "color": (0, 165, 255)}, # 橙色
}

# ===== 3. 打开摄像头 =====
cap = cv2.VideoCapture(0)

print("按 'q' 键退出")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("摄像头打开失败")
        break

    # 转换为 MediaPipe 需要的 RGB 格式
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 执行手势识别
    result = recognizer.recognize(mp_image)

    # ===== 4. 处理识别结果 =====
    # 默认值：没有识别到任何手势
    display_label = "No Gesture"
    display_color = (128, 128, 128)  # 灰色

    if result.gestures:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        score = top_gesture.score

        # 检查是否是我们要的 4 个手势之一
        if gesture_name in gesture_config:
            config = gesture_config[gesture_name]
            display_label = config["label"]
            display_color = config["color"]
            # 可选：在标签后面加上置信度
            # display_label = f"{config['label']} ({score:.2f})"
        else:
            # 识别到了手，但不是 4 个手势之一
            display_label = "Other"

    # ===== 5. 在画面上显示 =====
    # 大号文字显示手势名称
    cv2.putText(frame, display_label, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, display_color, 5)

    # 显示操作提示
    cv2.putText(frame, "Press 'q' to quit", (50, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # ===== 6. 显示画面 =====
    cv2.imshow('Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()