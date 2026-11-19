import os
import time
import argparse
import tempfile

# 避免 matplotlib 在沙盒里写缓存报 warning
os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import cv2
from ultralytics import YOLO


# ===================== 配置 =====================
DEFAULT_MODEL = "add_face.pt"   # 自训练 3 类模型: 0 close / 1 open / 2 face
CONF_THRESH = 0.5               # 置信度门槛：低于此值视为"看不清"，不执行指令

# 手势 → 指令 映射（open=张开手, close=握拳；语义由 ESP32 固件定义，可在此对调）
GESTURE_TO_CMD = {1: "1", 0: "0"}
CMD_TO_NAME = {"1": "开灯", "0": "关灯"}
CLASS_NAME = {0: "close(握拳)", 1: "open(张开手)", 2: "face(人脸)"}


def parse_args():
    p = argparse.ArgumentParser(description="手势控制智能家具 Agent")
    p.add_argument("--model", default=DEFAULT_MODEL, help="YOLO 模型路径")
    p.add_argument("--sim", action="store_true",
                   help="模拟模式：不连 MQTT，只在终端/画面打印指令（便于复现）")
    p.add_argument("--conf", type=float, default=CONF_THRESH, help="置信度门槛")
    p.add_argument("--require_face", action="store_true",
                   help="实验特性：要求检测到人脸才执行（默认关，因实测人脸框不稳定）")
    p.add_argument("--camera", type=int, default=0, help="摄像头编号")
    p.add_argument("--broker", default="broker.emqx.io", help="MQTT broker")
    p.add_argument("--port", type=int, default=1883)
    p.add_argument("--topic", default="esp32-client-123/helloworld")
    p.add_argument("--images", default=None, help="无头模式：指定图片文件夹，逐张识别")
    p.add_argument("--out", default="runs/demo", help="无头模式输出标注图目录")
    p.add_argument("--log", default=None, help="把运行日志同时写入该文件(便于事后慢慢看)")
    p.add_argument("--delay", type=float, default=0,
                   help="每帧/每张图后的延时(秒)；调大可放慢速度便于观察(建议 0.3~1)")
    return p.parse_args()


# ===================== MQTT（懒加载，sim/无头模式不需要） =====================
def make_mqtt_client(broker, port, topic, client_id):
    """连接 MQTT；失败返回 None（调用方降级为本地日志，不崩溃）。"""
    try:
        from paho.mqtt import client as mqtt_client
    except ImportError:
        print("[MQTT] 未安装 paho-mqtt，已降级为本地日志模式（pip install paho-mqtt 可启用）。")
        return None
    try:
        def on_connect(c, u, f, rc, props):
            print(f"[MQTT] 连接{'成功' if rc == 0 else '失败 rc=' + str(rc)}")
        client = mqtt_client.Client(
            client_id=client_id,
            callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
        )
        client.on_connect = on_connect
        client.connect(broker, port, keepalive=30)
        client.loop_start()
        return client
    except Exception as e:
        print(f"[MQTT] 连接异常：{e} — 降级为本地日志模式。")
        return None


def publish(client, topic, msg, log=print):
    if client is None:
        log(f"[本地日志] 拟发送指令 -> {msg} ({CMD_TO_NAME.get(msg, '?')})")
        return
    res = client.publish(topic, msg)
    ok = res[0] == 0
    log(f"[{'MQTT' if ok else 'MQTT失败'}] 发送 {msg} ({CMD_TO_NAME.get(msg, '?')})")


def make_logger(log_path):
    """返回一个 log(msg) 函数：同时打印到终端并(可选)写入文件。"""
    fh = open(log_path, "a", encoding="utf-8") if log_path else None
    def log(msg):
        print(msg)
        if fh:
            fh.write(msg + "\n")
            fh.flush()
    return log


# ===================== 核心：单帧决策（Agent 的"判断"一步） =====================
def decide(boxes, conf_thresh, require_face):
    """
    输入：YOLO 结果的 boxes（含 .cls 类别 与 .conf 置信度）
    输出：(command, status_text, status_color, info)
      command      : "0"/"1" 或 None（None=本帧不执行）
      status_text  : 给人看的状态说明（兜底/反馈）
    """
    if boxes is None or len(boxes.cls) == 0:
        return None, "待命：未检测到手势", (128, 128, 128), {}

    cls = boxes.cls.tolist()
    conf = boxes.conf.tolist()

    # 人脸（类2）仅作可选"在场"提示，不参与指令
    face_seen = 2 in cls and any(c == 2 and conf[i] >= conf_thresh
                                 for i, c in enumerate(cls))
    if require_face and not face_seen:
        return None, "待命：未检测到人，暂不执行", (0, 165, 255), {"face": face_seen}

    # 在达到置信度门槛的检测里，取置信度最高的手势类（0/1）
    best_cls, best_conf = None, 0.0
    gesture_confs = []  # 所有"手势类"的置信度（用于区分"没手势"与"手势但太低"）
    for i, c in enumerate(cls):
        if c in GESTURE_TO_CMD:
            gesture_confs.append(conf[i])
            if conf[i] >= conf_thresh and conf[i] > best_conf:
                best_cls, best_conf = int(c), conf[i]

    if best_cls is None:
        if gesture_confs:
            hi = max(gesture_confs)
            return None, f"低置信度，请重试 (手势最高 {hi:.2f}<{conf_thresh})", (0, 0, 255), {}
        return None, "待命：未识别到开/关手势", (128, 128, 128), {}

    cmd = GESTURE_TO_CMD[best_cls]
    return cmd, f"识别到 {CLASS_NAME[best_cls]} 置信度 {best_conf:.2f}", (0, 255, 0), \
        {"cls": best_cls, "conf": best_conf}


# ===================== 实时模式（摄像头） =====================
def run_live(args):
    model = YOLO(args.model)
    log = make_logger(args.log)
    client = None if args.sim else make_mqtt_client(
        args.broker, args.port, args.topic, "esp32-client-123")
    if not args.sim and client is None:
        log("[提示] 将以本地日志模式运行（不真正发送 MQTT）。")
    if not args.sim and client is not None:
        publish(client, args.topic, "手势识别端开始工作", log=log)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        log("错误：无法打开摄像头。")
        return

    device_state = None     # 当前设备状态 "0"/"1"
    current_cmd = None
    state_start = None
    last_sent = None
    REQUIRED_DURATION = 1.5  # 防抖：同一手势需持续这么久才发指令
    last_status = None      # 只在“状态文字”变化时打印，避免一帧一行刷屏
    paused = False

    log("摄像头已打开。按 'q' 退出，按 'p' 暂停/继续。")
    try:
        while True:
            if paused:
                # 暂停：停在上一次画面，叠加暂停提示，不做推理
                h = annotated.shape[0]
                cv2.putText(annotated, "[PAUSED] 按 p 继续", (20, h - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                cv2.imshow("手势控制智能家具 Agent", annotated)
                key = cv2.waitKey(max(1, int(args.delay * 1000))) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = False
                    log("[继续]")
                continue

            ret, frame = cap.read()
            if not ret:
                log("错误：读取画面失败。")
                break

            results = model(frame, verbose=False)
            boxes = results[0].boxes
            cmd, status, color, _ = decide(boxes, args.conf, args.require_face)

            # 防抖 + 状态变化才发送：避免抖动/重复发同一指令
            now = time.time()
            if cmd != current_cmd:
                current_cmd = cmd
                state_start = now
            elif cmd is not None and device_state != cmd:
                if now - state_start >= REQUIRED_DURATION:
                    publish(client, args.topic, cmd, log=log)
                    device_state = cmd
                    last_sent = cmd
                    state_start = now

            # 只在状态文字变化时打印一行，避免刷屏看不清
            if status != last_status:
                log(f"[状态] {status}  →  拟指令 {cmd}")
                last_status = status

            # 可视化（给人看 Agent 在做什么 = 人机协作/反馈）
            annotated = results[0].plot()
            h = annotated.shape[0]
            cv2.putText(annotated, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(annotated, f"设备状态: {CMD_TO_NAME.get(device_state, '--')}",
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.imshow("手势控制智能家具 Agent", annotated)
            key = cv2.waitKey(max(1, int(args.delay * 1000))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = True
                log("[暂停] 按 p 继续（画面已冻结，可慢慢看）")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if client is not None:
            client.loop_stop()


# ===================== 无头模式（图片文件夹，用于复现/演示） =====================
def run_images(args):
    model = YOLO(args.model)
    log = make_logger(args.log)
    os.makedirs(args.out, exist_ok=True)
    files = [f for f in os.listdir(args.images)
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    log(f"无头模式：在 {args.images} 找到 {len(files)} 张图片，"
        f"识别并保存标注图到 {args.out}")
    for f in sorted(files):
        path = os.path.join(args.images, f)
        img = cv2.imread(path)
        results = model(img, verbose=False)
        boxes = results[0].boxes
        cmd, status, color, _ = decide(boxes, args.conf, args.require_face)
        annotated = results[0].plot()
        cv2.putText(annotated, status, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.imwrite(os.path.join(args.out, f), annotated)
        log(f"  {f}: {status} -> 拟指令 {cmd}")
        if args.delay:
            time.sleep(args.delay)
    log("无头模式完成：这是流水线'识别→决策'的证明，可在无摄像头/无硬件环境下复现。")


def main():
    args = parse_args()
    if args.images:
        run_images(args)
    else:
        run_live(args)


if __name__ == "__main__":
    main()
