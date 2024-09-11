#include <Adafruit_MPU6050.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include "FS.h"
#include <ESP32Time.h>
#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>

#include <Adafruit_NeoPixel.h>

//#define LED_PIN 1
//Freenove_ESP32_WS2812 led = Freenove_ESP32_WS2812(1, LED_PIN, 0, TYPE_GRB);
Adafruit_NeoPixel rgb_led_1 = Adafruit_NeoPixel(1, 1, NEO_GRB + NEO_KHZ800);

Adafruit_MPU6050 mpu;
int prevAccTime;

WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP);
ESP32Time rtc;


const char* ssid = "ssid";
const char* password = "passwort";

void setLED(uint8_t r,uint8_t g,uint8_t b) {
  rgb_led_1.setPixelColor(0, rgb_led_1.Color(r, g, b));
  rgb_led_1.show();
}


#define BUFFER_SIZE 1000
struct SensorData {
  float accX;
  float accY;
  float accZ;
  float gyroX;
  float gyroY;
  float gyroZ;
  String timestamp;
};
SensorData buffer[BUFFER_SIZE];
int bufferIndex = 0;
unsigned long lastSaveTime = 0;

File Data;
SPIClass sdspi = SPIClass();
String filename;

void saveDataToSD() {
  Data = SD.open(filename, FILE_APPEND);
  if (Data) {
    for (int i = 0; i < bufferIndex; i++) {
      Data.print(buffer[i].accX);
      Data.print(",");
      Data.print(buffer[i].accY);
      Data.print(",");
      Data.print(buffer[i].accZ);
      Data.print(",");
      Data.print(buffer[i].gyroX);
      Data.print(",");
      Data.print(buffer[i].gyroY);
      Data.print(",");
      Data.print(buffer[i].gyroZ);
      Data.print(",");
      Data.println(buffer[i].timestamp);
    }
    Data.close();
  } else {
    Serial.println("Failed to open file for appending");
  }
  bufferIndex = 0;
}

void setup() {
  Serial.begin(115200);

  rgb_led_1.begin();
  rgb_led_1.setBrightness(10);
  //setLED(255,0,0);

  pinMode(SD_ENABLE, OUTPUT);
  digitalWrite(SD_ENABLE, LOW);
  sdspi.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI, VSPI_SS);
  if (!SD.begin(VSPI_SS, sdspi)) {
    Serial.println("failed to initialize SD card");
    while (1) {
      delay(10);
    }
  }

  if (!mpu.begin(0x68, &Wire1)) {
    Serial.println("failed to find MPU6050 chip");
    while (1) {
      delay(10);
    }
  }

  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG); // 
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);


  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("not connected");
  }

  //rtc.setTime(0);  // set time to 0 (unix epoch)

  timeClient.begin();
  timeClient.update();
  rtc.setTime(timeClient.getEpochTime());
  WiFi.disconnect();

  filename = "/data_" + rtc.getTime("%F_%H-%M-%S") + ".csv";
  Data = SD.open(filename, FILE_WRITE);
  if (!Data) {
    Serial.println("failed to create file");
    while (1) {
      delay(10);
    }
  }
  Data.print("AccX,AccY,AccZ,GyrX,GyrY,GyrZ,Time\n");
  Data.close();
  lastSaveTime = millis();
  prevAccTime = millis();


  // add confirmation that everything is set up
  setLED(0, 255, 0);

}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  String timestampString = "000" + String(rtc.getMillis());
  String timestamp = String(rtc.getEpoch()) + timestampString.substring(timestampString.length() - 3);

  buffer[bufferIndex].accX = a.acceleration.x;
  buffer[bufferIndex].accY = a.acceleration.y;
  buffer[bufferIndex].accZ = a.acceleration.z;
  buffer[bufferIndex].gyroX = g.gyro.x;
  buffer[bufferIndex].gyroY = g.gyro.y;
  buffer[bufferIndex].gyroZ = g.gyro.z;
  buffer[bufferIndex].timestamp = timestamp;
  bufferIndex++;
  
  // alle 10 sekunden -> speichern dauert 0.5sekunden
  if (bufferIndex >= BUFFER_SIZE || millis() - lastSaveTime >= 10000) {  // write every 10 seconds
    saveDataToSD();
    //Serial.println(timestamp);
    lastSaveTime = millis();
  }
  // warte bis 10 milli vorbei sind
  if((millis() - prevAccTime) < 10) {
    delay(10-(millis() - prevAccTime));
  }
  
  //Serial.println("acce: " + String(millis() - prevAccTime));
  prevAccTime = millis();
}
