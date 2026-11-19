import cv2
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from paho.mqtt import client as mqtt_client

# ===== MQTT 配置 =====
broker = 'broker.emqx.io'
port = 1883
topic = "esp32-client-123/helloworld"
client_id = 'esp32-client-123'

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("✅ Connected to MQTT Broker!")
        else:
            print(f"❌ Failed to connect, return code {rc}")

    client = mqtt_client.Client(
        client_id=client_id,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2
    )
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, msg):
    result = client.publish(topic, msg)
    if result[0] == 0:
        print(f"📤 Sent `{msg}` to topic `{topic}`")
    else:
        print(f"❌ Failed to send message to topic {topic}")

# ===== MediaPipe 手势识别器初始化 =====
model_path = 'gesture_recognizer.task'  # 确保文件在代码同目录

base_options = python.BaseOptions(
    model_asset_path=model_path,
    delegate=python.BaseOptions.Delegate.CPU  # Mac用户需要这行
)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ===== 手势 → MQTT 消息映射 =====
gesture_to_mqtt = {
    "Open_Palm":   "0",   # OPEN
    "Closed_Fist": "1",   # CLOSE
    "Thumb_Up":    "2",   # UP
    "Thumb_Down":  "3",   # DOWN
}

# ===== 状态跟踪变量 =====
last_sent = None
current_state = None
state_start_time = None
required_duration = 1.5  # 持续 1.5 秒才发送

# ===== 连接 MQTT =====
mqtt_client = connect_mqtt()
mqtt_client.loop_start()
publish(mqtt_client, "🖐️ 手势识别端开始工作")

# ===== 打开摄像头 =====
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

print("📷 摄像头已打开，按 'q' 键退出")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Error: Failed to capture image.")
        break

    # 转换颜色空间：BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 执行手势识别
    result = recognizer.recognize(mp_image)

    # ===== 识别当前手势 =====
    new_state = None
    if result.gestures:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        confidence = top_gesture.score

        # 检查是否是我们关心的 4 个手势之一
        if gesture_name in gesture_to_mqtt:
            new_state = gesture_to_mqtt[gesture_name]
            print(f"👋 识别到: {gesture_name} (置信度: {confidence:.2f})")

    # ===== 状态持续逻辑（1.5秒防抖） =====
    if new_state != current_state:
        # 状态变化，重置计时器
        current_state = new_state
        state_start_time = time.time()
    elif current_state is not None and current_state != last_sent:
        # 状态持续达到阈值，发送 MQTT
        if time.time() - state_start_time >= required_duration:
            publish(mqtt_client, current_state)
            last_sent = current_state

    # ===== 可视化：显示当前识别结果 =====
    display_text = "No Gesture"
    display_color = (128, 128, 128)  # 灰色

    if current_state is not None:
        # 反向查找手势名称（用于显示）
        for gesture_name, state_val in gesture_to_mqtt.items():
            if state_val == current_state:
                display_text = gesture_name
                break
        display_color = (0, 255, 0)  # 识别到目标手势显示绿色
    else:
        display_text = "No Gesture"

    # 在画面左上角显示大号文字
    cv2.putText(frame, display_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, display_color, 4)

    # 显示操作提示
    cv2.putText(frame, "Press 'q' to quit", (50, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 显示画面
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== 清理资源 =====
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
print("🛑 程序已退出")