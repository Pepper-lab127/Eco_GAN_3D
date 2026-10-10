#include <Servo.h>

// ---------------- Pins ----------------
const int TRIG_PIN = 7;
const int ECHO_PIN = 6;
const int SOUND_PIN = A1;
const int DISTANCE_SERVO_PIN = 8;
const int SOUND_SERVO_PIN = 9;

// ---------------- Distance feather ----------------
const float MAX_DISTANCE = 70;     // cm: anything farther away -> rest position
const float MIN_DISTANCE = 5;      // cm: this close (or closer) -> 180 degrees
const float DISTANCE_SMOOTHING = 0.2;  // 0..1: lower = smoother but slower

// ---------------- Sound feather ----------------
const int QUIET_LOUDNESS = 20;     // at or below this = quiet
const int LOUD_LOUDNESS  = 200;    // at or above this = full swing
const float FALL_SPEED = 40;       // degrees per second the feather sinks back after a sound

// ---------------- Both feathers ----------------
const int REST_ANGLE = 90;
const int MAX_ANGLE  = 180;
const int DEAD_BAND  = 2;          // only move a servo if its angle changes this much

const int SAMPLES = 5;             // distance readings per loop (the middle one is used)
const float NOTHING = 999;         // stands for "no echo"

Servo distanceFeather;
Servo soundFeather;

float distanceAngle = REST_ANGLE;  // eased angle for the distance feather
float soundAngle = REST_ANGLE;     // angle for the sound feather
int distanceServo = REST_ANGLE;    // angle last sent to each servo
int soundServo = REST_ANGLE;
unsigned long lastLoop = 0;

// ---------------- Sensor readings ----------------

// One distance reading in cm, or NOTHING if no echo came back
float readDistance() {
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  unsigned long echo = pulseIn(ECHO_PIN, HIGH, 25000UL);
  float cm = echo / 58.0;
  if (echo == 0 || cm < 2) return NOTHING;   // no echo, or a glitch
  return cm;
}

// Loudness: how much the microphone signal swings in the given time
int readLoudness(unsigned long ms) {
  int lowest = 1023;
  int highest = 0;
  unsigned long start = millis();
  while (millis() - start < ms) {
    int v = analogRead(SOUND_PIN);
    if (v < lowest) lowest = v;
    if (v > highest) highest = v;
  }
  return highest - lowest;
}

// Map a value from [fromLow, fromHigh] to an angle from REST_ANGLE to MAX_ANGLE
float toAngle(float value, float fromLow, float fromHigh) {
  float t = (value - fromLow) / (fromHigh - fromLow);
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  return REST_ANGLE + t * (MAX_ANGLE - REST_ANGLE);
}

void setup() {
  Serial.begin(9600);
  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);
  distanceFeather.attach(DISTANCE_SERVO_PIN);
  soundFeather.attach(SOUND_SERVO_PIN);
  distanceFeather.write(REST_ANGLE);
  soundFeather.write(REST_ANGLE);
  lastLoop = millis();
}

void loop() {
  // 1. Take 5 distance readings, and listen for sound in between them.
  //    Listening between pings means a short clap isn't missed while
  //    the distance sensor is busy.
  float r[SAMPLES];
  int loudness = 0;
  for (int i = 0; i < SAMPLES; i++) {
    r[i] = readDistance();
    int l = readLoudness(20);
    if (l > loudness) loudness = l;    // keep the loudest moment
  }

  // Sort the distance readings and use the middle one
  for (int i = 1; i < SAMPLES; i++) {
    float v = r[i];
    int j = i - 1;
    while (j >= 0 && r[j] > v) {
      r[j + 1] = r[j];
      j--;
    }
    r[j + 1] = v;
  }
  float distance = r[SAMPLES / 2];

  // How long this loop took, for the sound feather's fall speed
  unsigned long now = millis();
  float seconds = (now - lastLoop) / 1000.0;
  lastLoop = now;

  // 2. Distance feather: 70 cm -> 90 degrees ... 5 cm -> 180, eased
  float distanceTarget;
  if (distance > MAX_DISTANCE) {
    distanceTarget = REST_ANGLE;       // nothing within 70 cm
  } else {
    distanceTarget = toAngle(distance, MAX_DISTANCE, MIN_DISTANCE);
  }
  distanceAngle = distanceAngle + (distanceTarget - distanceAngle) * DISTANCE_SMOOTHING;

  // 3. Sound feather: jump up on a loud sound, then sink back slowly
  float soundTarget = toAngle(loudness, QUIET_LOUDNESS, LOUD_LOUDNESS);
  if (soundTarget > soundAngle) {
    soundAngle = soundTarget;
  } else {
    soundAngle = soundAngle - FALL_SPEED * seconds;
    if (soundAngle < soundTarget) soundAngle = soundTarget;
  }

  // 4. Move the servos, but only for a real change
  int a = (int)(distanceAngle + 0.5);
  if (abs(a - distanceServo) >= DEAD_BAND) {
    distanceFeather.write(a);
    distanceServo = a;
  }
  a = (int)(soundAngle + 0.5);
  if (abs(a - soundServo) >= DEAD_BAND) {
    soundFeather.write(a);
    soundServo = a;
  }

  // 5. Print what's happening
  Serial.print("distance: ");
  if (distance == NOTHING) Serial.print("nothing");
  else Serial.print(distance);
  Serial.print("  servo: ");
  Serial.print(distanceServo);
  Serial.print("   |   loudness: ");
  Serial.print(loudness);
  Serial.print("  servo: ");
  Serial.println(soundServo);
}
