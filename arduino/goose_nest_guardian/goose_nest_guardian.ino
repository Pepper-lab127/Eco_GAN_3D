/*
 * GOOSE NEST GUARDIAN (simple version)
 *
 * Reads the sensors and points each grass feather according to one
 * reading. No modes, no threat levels: every reading moves its own
 * feather between DOWN and UP.
 *
 *   Feather 0 (far left)   temperature   warmer   = higher
 *   Feather 1 (left)       humidity      wetter   = higher
 *   Feather 2 (center)     distance      closer   = higher
 *   Feather 3 (right)      light         brighter = higher
 *   Feather 4 (far right)  sound         louder   = higher
 *
 * Wiring is the same as in wiring.html. The motion sensors (D2, D3)
 * and the buzzer (A2) can stay connected; this version doesn't use them.
 *
 * Libraries: "DHT sensor library" by Adafruit (+ "Adafruit Unified
 * Sensor"), and Servo.
 */

#include <Servo.h>
#include <DHT.h>

// ---------------- Pins ----------------
const int PIN_DHT           = A3;
const int PIN_LIGHT         = A0;
const int PIN_SOUND         = A1;
const int PIN_US_LEFT_TRIG  = 7;
const int PIN_US_LEFT_ECHO  = 6;
const int PIN_US_RIGHT_TRIG = 5;
const int PIN_US_RIGHT_ECHO = 4;
const int PIN_SERVO[5] = {8, 9, 10, 11, 12};

// ---------------- Servo angles ----------------
// Angle where the grass is fully down, and fully up. If a servo is
// mounted the other way round, swap its two numbers.
const int DOWN_ANGLE[5] = { 20,  20,  20,  20,  20};
const int UP_ANGLE[5]   = {160, 160, 160, 160, 160};

// ---------------- Sensor ranges ----------------
// Each reading is mapped from its LOW value (feather down) to its
// HIGH value (feather up). Readings outside the range are clamped.
// Watch the Serial Monitor and adjust these to suit your room.
const float TEMP_LOW   = 18;    // deg C
const float TEMP_HIGH  = 32;
const float HUM_LOW    = 30;    // % relative humidity
const float HUM_HIGH   = 85;
const float DIST_FAR   = 150;   // cm: this far away or more = feather down
const float DIST_NEAR  = 10;    // cm: this close = feather up
const float LIGHT_LOW  = 0;     // % of the sensor's range
const float LIGHT_HIGH = 100;
const float SOUND_LOW  = 5;     // loudness (peak-to-peak reading)
const float SOUND_HIGH = 100;

// How much each new reading counts (0..1). Lower = smoother but slower.
const float SMOOTHING = 0.2;

// --------------------------------------
DHT dht(PIN_DHT, DHT11);
Servo feathers[5];

float temperature = 21;
float humidity    = 50;
float distance    = DIST_FAR;
float light       = 50;
float sound       = 0;

int lastAngle[5] = {-1, -1, -1, -1, -1};
unsigned long lastDhtRead = 0;
unsigned long lastPrint = 0;

// Map a value from [low, high] onto a servo's DOWN..UP angle.
int toAngle(int servo, float value, float low, float high) {
  float t = (value - low) / (high - low);   // 0 at low, 1 at high
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return DOWN_ANGLE[servo] + t * (UP_ANGLE[servo] - DOWN_ANGLE[servo]);
}

// Move a servo, but only if the angle changed by 2 degrees or more.
// This stops the servos buzzing from tiny changes in the readings.
void moveFeather(int servo, int angle) {
  if (abs(angle - lastAngle[servo]) >= 2) {
    feathers[servo].write(angle);
    lastAngle[servo] = angle;
  }
}

// Distance in cm from one HC-SR04, or -1 if there is no echo.
float readDistance(int trigPin, int echoPin) {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  unsigned long echo = pulseIn(echoPin, HIGH, 25000UL);
  float cm = echo / 58.0;
  if (echo == 0 || cm < 3) return -1;   // no echo, or a glitch
  return cm;
}

// Loudness: how far the microphone signal swings in 50 ms.
float readSound() {
  int lowest = 1023, highest = 0;
  unsigned long start = millis();
  while (millis() - start < 50) {
    int v = analogRead(PIN_SOUND);
    if (v < lowest) lowest = v;
    if (v > highest) highest = v;
  }
  return highest - lowest;
}

void setup() {
  Serial.begin(9600);
  dht.begin();
  pinMode(PIN_US_LEFT_TRIG, OUTPUT);
  pinMode(PIN_US_LEFT_ECHO, INPUT);
  pinMode(PIN_US_RIGHT_TRIG, OUTPUT);
  pinMode(PIN_US_RIGHT_ECHO, INPUT);
  for (int i = 0; i < 5; i++) {
    feathers[i].attach(PIN_SERVO[i]);
  }
  Serial.println(F("Goose Nest Guardian (simple version) started"));
}

void loop() {
  // --- Temperature and humidity: the DHT11 can only be read every 2 s ---
  if (millis() - lastDhtRead >= 2000) {
    lastDhtRead = millis();
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) temperature = t;   // keep the old value if a read fails
    if (!isnan(h)) humidity = h;
  }

  // --- Distance: the closer of the two sensors ---
  float left = readDistance(PIN_US_LEFT_TRIG, PIN_US_LEFT_ECHO);
  delay(30);                          // let the echo die down before the other sensor pings
  float right = readDistance(PIN_US_RIGHT_TRIG, PIN_US_RIGHT_ECHO);
  float nearest = DIST_FAR;           // nothing seen = far away
  if (left > 0 && left < nearest) nearest = left;
  if (right > 0 && right < nearest) nearest = right;
  distance = distance * (1 - SMOOTHING) + nearest * SMOOTHING;

  // --- Light and sound, smoothed so the feathers don't jitter ---
  float lightNow = analogRead(PIN_LIGHT) * 100.0 / 1023.0;
  light = light * (1 - SMOOTHING) + lightNow * SMOOTHING;
  sound = sound * (1 - SMOOTHING) + readSound() * SMOOTHING;

  // --- Point the feathers ---
  moveFeather(0, toAngle(0, temperature, TEMP_LOW, TEMP_HIGH));
  moveFeather(1, toAngle(1, humidity, HUM_LOW, HUM_HIGH));
  moveFeather(2, toAngle(2, distance, DIST_FAR, DIST_NEAR));  // far -> down, near -> up
  moveFeather(3, toAngle(3, light, LIGHT_LOW, LIGHT_HIGH));
  moveFeather(4, toAngle(4, sound, SOUND_LOW, SOUND_HIGH));

  // --- Show the readings twice a second ---
  if (millis() - lastPrint >= 500) {
    lastPrint = millis();
    Serial.print(F("temp="));        Serial.print(temperature, 0);
    Serial.print(F("C  humidity=")); Serial.print(humidity, 0);
    Serial.print(F("%  distance=")); Serial.print(distance, 0);
    Serial.print(F("cm  light="));   Serial.print(light, 0);
    Serial.print(F("%  sound="));    Serial.print(sound, 0);
    Serial.print(F("   angles:"));
    for (int i = 0; i < 5; i++) { Serial.print(' '); Serial.print(lastAngle[i]); }
    Serial.println();
  }
}
