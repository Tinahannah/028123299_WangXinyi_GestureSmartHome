# 基于手势识别的智能家居控制系统

一个可运行的端到端 Agent：摄像头采集画面 → YOLO11n 识别手势 → 置信度门槛与防抖判定 → MQTT 发送开/关指令 → ESP32 驱动灯具，并附带评估脚本与无头复现模式。

## 一、项目结构

```
.
├── agent.py                     端到端 Agent 主程序（识别—决策—通信—反馈）
├── train.py                     YOLO11n 训练脚本
├── evaluate.py                  测试集评估脚本（输出 mAP / Precision / Recall）
├── predict.py                   早期识别脚本（仅摄像头显示，保留作参考）
├── predictsendmqtt.py           早期识别 + MQTT 脚本（agent.py 的前身，保留作参考）
├── close_eval.yaml              评估数据配置（3 类：close/open/face）
├── close.yaml                   原始训练数据配置
├── requirements.txt             Python 依赖清单
├── add_face.pt                  部署权重（3 类：close/open/face）
├── total.pt                     早期权重（2 类：close/open）
├── open-close-datasets/         手势数据集（train 129 / val 16 / test 16）
│   ├── train/{images,labels}
│   ├── val/{images,labels}
│   └── test/{images,labels}
├── gesture-21/                  MediaPipe 手势识别对照实现（基线）
│   ├── 1.py / 2.py              采集与识别辅助脚本
│   ├── predict.py               MediaPipe 识别主程序
│   ├── predictmqtt(old).py      旧版 MQTT 脚本
│   └── gesture_recognizer.task  MediaPipe 预训练模型
├── runs/                        运行产物
│   ├── demo/                    无头模式生成的标注演示图（16 张）
│   ├── detect/                  evaluate.py 评估时生成的检测图（含 val~val10）
│   ├── agent_log.txt            运行日志示例
│   └── fontlist-v330.json       OpenCV 中文字体缓存文件
├── esp32/                       ESP32 接收端程序
│   └── esp32_light.ino          订阅 MQTT 指令并驱动灯具（WiFi 信息留空）
├── 实际演示视频.mp4              真实硬件演示（手势 + ESP32 + 灯具）
├── 模拟演示视频.mp4              模拟模式演示（识别框 + 终端日志）
└── README.md                    本说明文件
```

## 二、环境与依赖

- 操作系统：macOS（Apple M3），其他平台亦可运行
- Python：3.9+
- 核心依赖：`ultralytics>=8.3`、`opencv-python>=4.7`、`paho-mqtt>=2.0`、`torch>=2.0`、`numpy>=1.24`

安装依赖：

```bash
pip install -r requirements.txt
```

## 三、运行方式

在装有上述依赖的 Python 环境中，进入项目根目录后执行。

### 1. 真实硬件模式（连接 ESP32）

```bash
python agent.py
```

识别到张开手持续约 1.5 秒后发送 `"1"`（开灯），握拳发送 `"0"`（关灯），通过 MQTT 发布到 `broker.emqx.io` 的 `esp32-client-123/helloworld` 主题。

### 2. 模拟模式（无需硬件，推荐演示用）

```bash
python agent.py --sim
```

不连接 MQTT，仅在终端打印拟发送指令，画面仍实时显示识别框与设备状态。需要摄像头和屏幕窗口。

### 3. 无头复现模式（无需摄像头与屏幕）

```bash
python agent.py --images open-close-datasets/test/images --out runs/demo
```

逐张识别测试图片，输出标注图到 `runs/demo`。评分者仅凭代码和测试集即可复现"识别—决策"核心流程。

### 4. 测试集评估

```bash
python evaluate.py
```

输出 mAP@0.5、mAP@0.5:0.95、Precision、Recall 及每类指标。

### 5. ESP32 接收端烧录

用 Arduino IDE 打开 `esp32/esp32_light.ino`，在文件开头填入自己的 WiFi 名称和密码后烧录到 ESP32。程序会连接 `broker.emqx.io` 并订阅 `esp32-client-123/helloworld` 主题：收到 `"1"` 点亮灯具，收到 `"0"` 熄灭灯具，与 `agent.py` 发送的指令一致。

> 说明：代码中的 `emqx` / `public` 是 EMQX 公共测试 Broker 的公开演示账号，不是个人隐私；WiFi 名称与密码在提交版本中已留空。

### 常用可选参数

| 参数                                | 作用                  |
| --------------------------------- | ------------------- |
| `--conf 0.5`                      | 调整置信度门槛，默认 0.5      |
| `--delay 0.3`                     | 放慢输出速度，便于观察         |
| `--log runs/agent_log.txt`        | 日志同时写入文件，便于回看       |
| `--require_face`                  | 开启人脸待命安全闸（可选，实测不稳定） |
| `--camera 0`                      | 指定摄像头编号             |
| `--broker` / `--port` / `--topic` | 自定义 MQTT 连接参数       |

## 四、手势与指令映射

| 类别    | 手势  | 指令        | 是否默认启用 |
| ----- | --- | --------- | ------ |
| open  | 张开手 | `1`（开灯）   | 是      |
| close | 握拳  | `0`（关灯）   | 是      |
| face  | 人脸  | 无（可选安全信号） | 否      |

只有置信度不低于门槛且持续约 1.5 秒的手势才触发指令；低置信度提示重试，无手势进入待命，MQTT 失败降级为本地日志。

## 五、数据来源与许可证

- **手势图像（close/open）**：本项目使用摄像头自行采集与标注，共 161 张（train 129 / val 16 / test 16）。
- **人脸图像（face）**：取自公开数据集 CASIA-FaceV5，由中国科学院自动化研究所发布，经飞桨 AI Studio 获取（<https://aistudio.baidu.com/datasetdetail/47943>）

| 组件                  | 来源                     | 许可证           |
| ------------------- | ---------------------- | ------------- |
| Ultralytics YOLO11n | ultralytics.com / PyPI | AGPL-3.0      |
| OpenCV              | opencv.org / PyPI      | Apache-2.0    |
| MediaPipe           | Google AI Edge         | Apache-2.0    |
| paho-mqtt           | Eclipse / PyPI         | EPL-2.0 / EDL |
| PyTorch             | pytorch.org            | BSD-3-Clause  |

具体条款以各项目官方最新声明为准。

## 六、实验结果摘要

在 test 集（16 张图像 / 14 个实例）上对 `add_face.pt` 评估：

| 指标           | 整体    | close | open  | face  |
| ------------ | ----- | ----- | ----- | ----- |
| mAP@0.5      | 0.995 | 0.995 | 0.995 | 0.995 |
| mAP@0.5:0.95 | 0.848 | 0.852 | 0.889 | 0.802 |
| Precision    | 0.988 | 0.992 | 0.983 | 0.988 |
| Recall       | 1.000 | 1.000 | 1.000 | 1.000 |

CPU（Apple M3）单帧推理约 34–45 ms，满足实时交互需求。

## 七、注意事项

- `face` 类在测试集指标良好，但真实摄像头下受光照与姿态影响常无法稳定框出，因此默认关闭人脸安全闸。
- 演示视频一真一模拟：真实视频展示硬件闭环，模拟视频展示识别与决策日志。
