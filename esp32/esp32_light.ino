/*
 * 手势控制智能家居 —— ESP32 接收端程序
 *
 * 连接 WiFi 后订阅 MQTT 主题 esp32-client-123/helloworld：
 *   收到 '1' → 开灯（板载 LED 点亮，D4/D5 置低电平）
 *   收到 '0' → 关灯（板载 LED 熄灭，D4/D5 置高电平）
 * 与 PC 端 agent.py 中 open→"1"、close→"0" 的指令映射一致。
 *
 * 烧录前请在下方 ssid / password 中填入自己的 WiFi 名称与密码。
 * 提交版本保持为空，避免泄露个人网络信息。
 */

#include <WiFi.h>
#include <PubSubClient.h>
#define LED_BUILTIN 2
#define LED_D4 4
#define LED_D5 5

// WiFi 配置（使用时自行填写）
const char *ssid = "";
const char *password = "";

// MQTT Broker：EMQX 公共测试服务器
const char *mqtt_broker = "broker.emqx.io";
const char *topic = "esp32-client-123/helloworld";
const char *mqtt_username = "emqx";    // 公共 Broker 的公开账号，非个人隐私
const char *mqtt_password = "public";  // 公共 Broker 的公开密码，非个人隐私
const int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);

void setup() {
 // 初始化引脚为输出模式
 pinMode(LED_BUILTIN, OUTPUT);
 pinMode(LED_D4, OUTPUT);
 digitalWrite(4, HIGH);
 pinMode(LED_D5, OUTPUT);
 digitalWrite(5, HIGH);
 // 串口波特率 115200，便于在串口监视器观察连接状态
 Serial.begin(115200);
 // 连接 WiFi 网络
 WiFi.begin(ssid, password);
 while (WiFi.status() != WL_CONNECTED) {
     delay(500);
     Serial.println("Connecting to WiFi..");
 }
 Serial.println("Connected to the WiFi network");
 // 连接 MQTT Broker 并注册回调函数
 client.setServer(mqtt_broker, mqtt_port);
 client.setCallback(callback);
 while (!client.connected()) {
     String client_id = "esp32-client-123";
     client_id += String(WiFi.macAddress());
     Serial.printf("The client %s connects to the public mqtt broker\n", client_id.c_str());
     if (client.connect(client_id.c_str(), mqtt_username, mqtt_password)) {
         Serial.println("Public emqx mqtt broker connected");
     } else {
         Serial.print("failed with state ");
         Serial.print(client.state());
         delay(2000);
     }
 }
 // 发布上线消息并订阅控制主题
 client.publish(topic, "Hi EMQ X I'm ESP32 ^^");
 client.subscribe(topic);
}

void callback(char *topic, byte *payload, unsigned int length) {
 Serial.print("Message arrived in topic: ");
 Serial.println(topic);
 Serial.print("Message:");
 if(length > 0 && ((char) payload[0]) == '1'){
    digitalWrite(LED_BUILTIN, HIGH);
    digitalWrite(LED_D4, LOW);
    digitalWrite(LED_D5, LOW);
 } else if (length > 0 && ((char) payload[0]) == '0'){
    digitalWrite(LED_BUILTIN, LOW);
    digitalWrite(LED_D4, HIGH);
    digitalWrite(LED_D5, HIGH);
 }
 Serial.println();
 Serial.println("-----------------------");
}

void loop() {
 client.loop();
}
