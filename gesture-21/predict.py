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
model_path = 'gesture_recognizer.task'

base_options = python.BaseOptions(
    model_asset_path=model_path,
    delegate=python.BaseOptions.Delegate.CPU
)
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# ===== 手势 → MQTT 消息映射（已对调） =====
gesture_to_mqtt = {
    "Open_Palm":   "1",   # 关灯
    "Closed_Fist": "0",   # 开灯
    "Thumb_Up":    "2",   # 拉开窗帘
    "Thumb_Down":  "3",   # 关上窗帘
}

# ===== 两组互锁状态 =====
light_state = None     # 灯：None / "0" / "1"
curtain_state = None   # 窗帘：None / "2" / "3"

# ===== 状态持续逻辑 =====
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

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = recognizer.recognize(mp_image)

    # ===== 识别当前手势 =====
    new_state = None
    if result.gestures:
        top_gesture = result.gestures[0][0]
        gesture_name = top_gesture.category_name
        confidence = top_gesture.score

        if gesture_name in gesture_to_mqtt:
            new_state = gesture_to_mqtt[gesture_name]
            print(f"👋 识别到: {gesture_name} (置信度: {confidence:.2f})")

    # ===== 状态持续逻辑（1.5秒防抖） =====
    if new_state != current_state:
        current_state = new_state
        state_start_time = time.time()
    elif current_state is not None:
        if time.time() - state_start_time >= required_duration:
            should_send = False

            if current_state == "0":       # 开灯
                if light_state != "0":
                    should_send = True
                    light_state = "0"
            elif current_state == "1":     # 关灯
                if light_state != "1":
                    should_send = True
                    light_state = "1"
            elif current_state == "2":     # 拉开窗帘
                if curtain_state != "2":
                    should_send = True
                    curtain_state = "2"
            elif current_state == "3":     # 关上窗帘
                if curtain_state != "3":
                    should_send = True
                    curtain_state = "3"

            if should_send:
                publish(mqtt_client, current_state)
                last_sent = current_state
                state_start_time = time.time()
            else:
                state_start_time = time.time()

    # ===== 可视化 =====
    display_text = "No Gesture"
    display_color = (128, 128, 128)

    if current_state is not None:
        for gesture_name, state_val in gesture_to_mqtt.items():
            if state_val == current_state:
                display_text = gesture_name
                break
        display_color = (0, 255, 0)
    else:
        display_text = "No Gesture"

    cv2.putText(frame, display_text, (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, display_color, 4)

    light_text = f"Light: {light_state if light_state else '--'}"
    curtain_text = f"Curtain: {curtain_state if curtain_state else '--'}"
    cv2.putText(frame, light_text, (50, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
    cv2.putText(frame, curtain_text, (50, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

    cv2.putText(frame, "Press 'q' to quit", (50, frame.shape[0] - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ===== 清理资源 =====
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()
print("🛑 程序已退出")