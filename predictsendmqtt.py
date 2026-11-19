from ultralytics import YOLO
import cv2
from paho.mqtt import client as mqtt_client
import time

# MQTT Configuration
broker = 'broker.emqx.io'
port = 1883
topic = "esp32-client-123/helloworld"
client_id = 'esp32-client-123'

# Initialize MQTT client
def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print(f"Failed to connect, return code {rc}\n")

    client = mqtt_client.Client(client_id=client_id, 
                               callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.connect(broker, port)
    return client

def publish(client, msg):
    result = client.publish(topic, msg)
    if result[0] == 0:
        print(f"Sent `{msg}` to topic `{topic}`")
    else:
        print(f"Failed to send message to topic {topic}")

# Load YOLO model
model = YOLO("./add_face.pt")

# Connect to MQTT
mqtt_client = connect_mqtt()
mqtt_client.loop_start()

# Open camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

publish(mqtt_client, "识别端开始工作。")

# State tracking variables
last_sent = None
current_state = None
state_start_time = None
required_duration = 1.5  # seconds

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    results = model(frame)
    
    # Print processing info (optional)
    print(results[0].speed)  # Show preprocess/inference/postprocess times
    print(results[0].boxes.conf)  # Show detection confidence
    
    # Get detected classes (assuming 0 and 1 are your class IDs)
    detected_classes = results[0].boxes.cls.tolist() if results[0].boxes.cls is not None else []
    
    # Determine current detection
    new_state = None
    if 1 in detected_classes:
        new_state = "1"
    elif 0 in detected_classes:
        new_state = "0"
    
    # State tracking logic
    if new_state != current_state:
        # State changed, reset timer
        current_state = new_state
        state_start_time = time.time()
    elif current_state is not None and current_state != last_sent:
        # Check if we've maintained this state for required duration
        if time.time() - state_start_time >= required_duration:
            publish(mqtt_client, current_state)
            print("消息发送成功！")
            last_sent = current_state
    
    # Visualize and display
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Pose Estimation", annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
mqtt_client.loop_stop()