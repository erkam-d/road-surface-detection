// erforderliche Bibliotheken einbinden
#include <Adafruit_MPU6050.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include "FS.h"
#include <ESP32Time.h>
#include <WiFi.h>
#include <NTPClient.h>
#include <WiFiUdp.h>

//#include <Adafruit_NeoPixel.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "road_surface_lstm_model_data.h"  // einbinden der TensorFlow Lite Modelldaten

// LED-Konfiguration
//Adafruit_NeoPixel rgb_led_1 = Adafruit_NeoPixel(1, 1, NEO_GRB + NEO_KHZ800);

// MPU6050 Sensor-Objekt
Adafruit_MPU6050 mpu;

// WLAN-zugangsdaten
const char* ssid = "ssid";
const char* password = "passwort";

// WLAN und zeit-Setup
WiFiUDP ntpUDP;
NTPClient timeClient(ntpUDP);
ESP32Time rtc;

// tensorflow Lite flobale variablen
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// TensorFlow Lite Arbeitsspeicher
// vllt durch RecordingMicroInterpreter besseren wert finden
constexpr int kTensorArenaSize = 8 * 1024;
uint8_t tensorArena[kTensorArenaSize];

// klassenlabels für die vorhersagen
const char* labels[] = {"asphalt", "cobblestone", "paving_stone"};

// puffer für vorhersagedaten
#define BUFFER_SIZE 440 // trial und error
struct PredictionData {
  String prediction;
  String timestamp;
};
PredictionData buffer[BUFFER_SIZE];
int bufferIndex = 0;
unsigned long lastSaveTime = 0;

// SD
File Data;
SPIClass sdspi = SPIClass();
String filename;

// Funktion zum Setzen der LED-Farbe
/*void setLED(uint8_t r, uint8_t g, uint8_t b) {
  rgb_led_1.setPixelColor(0, rgb_led_1.Color(r, g, b));
  rgb_led_1.show();
}*/

// funktion zum Speichern der Daten auf der SD-Karte
void saveDataToSD() {
  Data = SD.open(filename, FILE_APPEND);
  if (Data) {
    for (int i = 0; i < bufferIndex; i++) {
      Data.print(buffer[i].prediction);
      Data.print(",");
      Data.println(buffer[i].timestamp);
    }
    Data.close();
  } else {
    Serial.println("fehler beim Öffnen der Datei zum Anhängen");
  }
  bufferIndex = 0;
}

void setup() {
  Serial.begin(115200);
  Serial.println("starting setup");

  // LED-Initialisierung
  //rgb_led_1.begin();
  //rgb_led_1.setBrightness(10);
  //setLED(0, 0, 255);  // Blau während des Setups

  // SD-karten-initialisierung
  pinMode(SD_ENABLE, OUTPUT);
  digitalWrite(SD_ENABLE, LOW);
  sdspi.begin(VSPI_SCLK, VSPI_MISO, VSPI_MOSI, VSPI_SS);
  if (!SD.begin(VSPI_SS, sdspi)) {
    Serial.println("fehler bei der Initialisierung der SD-Karte");
    while (1) { delay(10); }
  }

  // MPU6050 densor-initialisierung
  if (!mpu.begin(0x68, &Wire1)) {
    Serial.println("MPU6050-Chip nicht gefunden");
    while (1) { delay(10); }
  }
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_250_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);

  // WLAN und zeit-initialisierung
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("nicht verbunden");
  }
  timeClient.begin();
  timeClient.update();
  rtc.setTime(timeClient.getEpochTime());
  WiFi.disconnect();

  // dateinamen für die Datenprotokollierung erstellen
  filename = "/data_" + rtc.getTime("%F_%H-%M-%S") + ".csv";
  Data = SD.open(filename, FILE_WRITE);
  if (!Data) {
    Serial.println("fehler beim erstellen der datei");
    while (1) { delay(10); }
  }
  Data.print("Prediction,Time\n");
  Data.close();
  lastSaveTime = millis();

  // tensorflow Lite modell-initialisierung
  tflModel = tflite::GetModel(model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Modellschema stimmt nicht überein!");
    while (1) { delay(10); }
  }
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, kTensorArenaSize, &tflErrorReporter);
  tflInterpreter->AllocateTensors();
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  //setLED(0, 255, 0);  // grün nach abschluss des setups
}

void loop() {
  // sensordaten abrufen
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // ingabe für das Modell vorbereiten
  tflInputTensor->data.f[0] = a.acceleration.x;
  tflInputTensor->data.f[1] = a.acceleration.y;
  tflInputTensor->data.f[2] = a.acceleration.z;
  tflInputTensor->data.f[3] = g.gyro.x;
  tflInputTensor->data.f[4] = g.gyro.y;
  tflInputTensor->data.f[5] = g.gyro.z;

  // inferenz durchführen
  if (tflInterpreter->Invoke() != kTfLiteOk) {
    Serial.println("Fehler beim Aufrufen des TensorFlow Lite-Modells");
    return;
  }

  // ergebnis mit der höchsten Wahrscheinlichkeit finden
  float* results = tflOutputTensor->data.f;
  int maxIndex = 0;
  for (int i = 1; i < 3; i++) {
    if (results[i] > results[maxIndex]) {
      maxIndex = i;
    }
  }

  // vorhersage und Zeitstempel im Puffer speichern
  String timestampString = "000" + String(rtc.getMillis());
  String timestamp = String(rtc.getEpoch()) + timestampString.substring(timestampString.length() - 3);

  buffer[bufferIndex].prediction = labels[maxIndex];
  buffer[bufferIndex].timestamp = timestamp;
  bufferIndex++;

  // vorhersage auf der seriellen konsole ausgeben
  // Serial.println("Prediction: " + String(labels[maxIndex]) + ", Timestamp: " + timestamp);

  // periodisches Speichern auf SD-Karte
  if (bufferIndex >= BUFFER_SIZE || millis() - lastSaveTime >= 10000) {  // alle 10 Sekunden schreiben - buffer allerdings kleiner
    saveDataToSD();
    lastSaveTime = millis();
  }

  delay(100);  // verzögerung
}
