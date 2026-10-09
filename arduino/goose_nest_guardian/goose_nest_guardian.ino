/*
 * ===================================================================
 *  GOOSE NEST GUARDIAN
 * ===================================================================
 *  A nest that watches its surroundings to judge whether they suit a
 *  Canada goose egg, and guards the egg from anything that comes near.
 *
 *  Five servos carry long blades of grass in a peacock-tail fan:
 *
 *            [1]  [2]  [3]
 *        [0]               [4]
 *                 (nest)
 *
 *    0 = far left   -> temperature
 *    1 = left       -> humidity
 *    2 = center     -> overall comfort
 *    3 = right      -> light
 *    4 = far right  -> air quality
 *
 *  MOOD MODE (nothing nearby)
 *    Each feather's height shows how good its factor is: a tall
 *    feather means the factor is ideal and a drooping one means it
 *    is poor. The motion shows the overall mood:
 *      CONTENT    -> slow, gentle ripple across the fan
 *      UNEASY     -> the worst feather twitches now and then
 *      DISTRESSED -> the whole fan sags and shivers
 *
 *  DEFENSE MODE (something approaches)
 *    ALERT  -> motion seen far away: the fan rises, leans toward the
 *              intruder and quivers
 *    WARN   -> something is getting close or coming fast: the fan
 *              spreads fully, rattles and hisses (if a buzzer is fitted)
 *    DEFEND -> something is right at the nest: the feathers on the
 *              intruder's side swat at it hard and fast
 *
 *  ------------------------------------------------------------------
 *  HARDWARE (Arduino Uno / Nano / Mega)
 *  ------------------------------------------------------------------
 *    DHT22 temperature/humidity  -> D2
 *    PIR motion, LEFT            -> D4
 *    PIR motion, CENTER          -> D5
 *    PIR motion, RIGHT           -> D6
 *    HC-SR04 ultrasonic TRIG     -> D7   (faces straight out, center)
 *    HC-SR04 ultrasonic ECHO     -> D8
 *    Servo feathers 0..4         -> D3, D9, D10, D11, D12
 *    Photoresistor (LDR)         -> A0   (LDR to 5V, 10k resistor to GND,
 *                                         junction to A0: brighter = higher)
 *    MQ-135 air quality (AO)     -> A1
 *    Passive buzzer (optional)   -> A2   (used as a digital pin to hiss)
 *
 *  !! Power the servos from a separate 5-6 V supply (at least 2-3 A)
 *  !! and connect its GND to the Arduino GND. Five servos swatting at
 *  !! once will brown out and reset the Arduino if run from its 5V pin.
 *
 *  LIBRARIES (Arduino IDE -> Library Manager)
 *    - "DHT sensor library" by Adafruit
 *    - "Adafruit Unified Sensor" (needed by the DHT library)
 *    - Servo (comes with the IDE)
 * ===================================================================
 */

#include <Servo.h>
#include <DHT.h>

// ===================================================================
//  PINS
// ===================================================================
const uint8_t PIN_DHT        = 2;
const uint8_t PIN_PIR_LEFT   = 4;
const uint8_t PIN_PIR_CENTER = 5;
const uint8_t PIN_PIR_RIGHT  = 6;
const uint8_t PIN_US_TRIG    = 7;
const uint8_t PIN_US_ECHO    = 8;
const uint8_t PIN_LIGHT      = A0;
const uint8_t PIN_AIR        = A1;
const uint8_t PIN_BUZZER     = A2;

const uint8_t NUM_FEATHERS = 5;
const uint8_t PIN_FEATHER[NUM_FEATHERS] = {3, 9, 10, 11, 12};

#define DHT_TYPE DHT22
#define USE_BUZZER 1          // set to 0 if you have no buzzer

// ===================================================================
//  SERVO CALIBRATION
//  For each feather, set the servo angle where the grass is fully
//  drooped/folded (DOWN) and fully raised/fanned out (UP). If a servo
//  is mounted mirrored, swap its two numbers. Start with a narrow
//  range and widen it so the grass never hits anything.
// ===================================================================
const int FEATHER_DOWN_ANGLE[NUM_FEATHERS] = { 20,  20,  20,  20,  20};
const int FEATHER_UP_ANGLE[NUM_FEATHERS]   = {160, 160, 160, 160, 160};

// ===================================================================
//  EGG ENVIRONMENT TARGETS
//  Each factor scores 1.0 inside its ideal range and falls to 0.0 at
//  the "bad" limits. Values are for incubating Canada goose eggs
//  (28-day incubation). Change them to suit your setup.
// ===================================================================

// Set to true for the last 3 days before hatching ("lockdown"), when
// humidity should be raised and the eggs are no longer turned.
const bool LOCKDOWN_PHASE = false;

// Temperature (deg C): goose eggs like about 37.5 C (99.5 F)
const float TEMP_BAD_LOW   = 35.0;
const float TEMP_IDEAL_LOW = 37.2;
const float TEMP_IDEAL_HI  = 37.8;
const float TEMP_BAD_HIGH  = 39.5;

// Relative humidity (%): around 50-55 % normally, 70-80 % at lockdown
const float HUM_BAD_LOW    = LOCKDOWN_PHASE ? 55.0 : 30.0;
const float HUM_IDEAL_LOW  = LOCKDOWN_PHASE ? 70.0 : 50.0;
const float HUM_IDEAL_HI   = LOCKDOWN_PHASE ? 80.0 : 60.0;
const float HUM_BAD_HIGH   = LOCKDOWN_PHASE ? 90.0 : 75.0;

// Light (0-100 % of the LDR range): a shaded nest is good. Strong
// direct sun overheats the egg and makes the nest easy to see.
const float LIGHT_BAD_LOW   = -1.0;   // darkness is fine
const float LIGHT_IDEAL_LOW =  0.0;
const float LIGHT_IDEAL_HI  = 40.0;
const float LIGHT_BAD_HIGH  = 85.0;

// Air quality (raw MQ-135 reading, 0-1023, higher = worse). These
// numbers are rough: watch the Serial Monitor in clean outdoor air and
// set AIR_IDEAL_HI a little above what you see there.
const float AIR_BAD_LOW   = -1.0;
const float AIR_IDEAL_LOW =  0.0;
const float AIR_IDEAL_HI  = 300.0;
const float AIR_BAD_HIGH  = 600.0;

// How much each factor counts toward the overall comfort score
enum Factor { F_TEMP, F_HUM, F_LIGHT, F_AIR, NUM_FACTORS };
const float FACTOR_WEIGHT[NUM_FACTORS] = {0.40, 0.30, 0.15, 0.15};
const char* const FACTOR_NAME[NUM_FACTORS] = {"temp", "humidity", "light", "air"};

// Which feather shows which factor (-1 = the overall score)
const int8_t FEATHER_FACTOR[NUM_FEATHERS] = {F_TEMP, F_HUM, -1, F_LIGHT, F_AIR};

// Mood thresholds on the overall score (0..1)
const float MOOD_CONTENT_MIN = 0.75;
const float MOOD_UNEASY_MIN  = 0.45;

// ===================================================================
//  DEFENSE SETTINGS
// ===================================================================
const float ALERT_DIST_CM       = 200.0;  // notice things inside this range
const float WARN_DIST_CM        = 100.0;  // threat display inside this range
const float DEFEND_DIST_CM      =  40.0;  // swat at things inside this range
const float APPROACH_FAST_CM_S  =  40.0;  // closing faster than this = WARN
const unsigned long MOTION_PERSIST_MS = 4000;  // lingering motion -> WARN
const unsigned long THREAT_HOLD_MS    = 4000;  // calm down one level after this

// ===================================================================
//  TIMING
// ===================================================================
const unsigned long PIR_WARMUP_MS      = 30000;  // PIRs give false hits at power-up
const unsigned long AIR_WARMUP_MS      = 60000;  // MQ-135 heater needs to warm up
const unsigned long DHT_INTERVAL_MS    = 2000;   // DHT22 can't be read faster
const unsigned long ANALOG_INTERVAL_MS = 200;
const unsigned long SONAR_INTERVAL_MS  = 70;
const unsigned long ANIM_INTERVAL_MS   = 20;
const unsigned long REPORT_INTERVAL_MS = 2000;

// ===================================================================
//  STATE
// ===================================================================
DHT dht(PIN_DHT, DHT_TYPE);

struct Feather {
  Servo servo;
  float pos;      // current position: 0 = drooped, 1 = fully raised
  float target;   // where it is heading
  float speed;    // how fast it moves, in positions per second
  int lastAngle;
};
Feather feathers[NUM_FEATHERS];

enum Mood { MOOD_CONTENT, MOOD_UNEASY, MOOD_DISTRESSED };
const char* const MOOD_NAME[] = {"CONTENT", "UNEASY", "DISTRESSED"};

enum ThreatLevel { THREAT_NONE, THREAT_ALERT, THREAT_WARN, THREAT_DEFEND };
const char* const THREAT_NAME[] = {"none", "ALERT", "WARN", "DEFEND"};

// Environment readings
float tempC = NAN;
float humidity = NAN;
float lightPct = 0;
float airRaw = 0;
bool  analogPrimed = false;

float factorScore[NUM_FACTORS];
bool  factorValid[NUM_FACTORS];
float overallScore = 0.5;
Mood  mood = MOOD_UNEASY;

// Threat sensing
bool  pirLeft = false, pirCenter = false, pirRight = false;
float distanceCm = -1;         // -1 = nothing in range
float approachSpeed = 0;       // cm/s, positive = getting closer
float sonarHistory[3] = {-1, -1, -1};
uint8_t sonarIndex = 0;
unsigned long motionStartMs = 0;

ThreatLevel threat = THREAT_NONE;
int8_t threatDir = 0;          // -1 = left, 0 = center, +1 = right
unsigned long threatChangedMs = 0;

// Uneasy twitch
int8_t twitchFeather = -1;
unsigned long twitchUntilMs = 0;
unsigned long nextTwitchMs = 0;

// Buzzer
bool hissing = false;
unsigned long lastHissChangeMs = 0;

// Timers
unsigned long lastDhtMs = 0, lastAnalogMs = 0, lastSonarMs = 0;
unsigned long lastAnimMs = 0, lastReportMs = 0;
unsigned long bootMs = 0;

// ===================================================================
//  HELPERS
// ===================================================================

// Score 1.0 inside [idealLow, idealHigh], falling linearly to 0.0 at the bad limits.
float rangeScore(float v, float badLow, float idealLow, float idealHigh, float badHigh) {
  if (v >= idealLow && v <= idealHigh) return 1.0;
  if (v <= badLow || v >= badHigh) return 0.0;
  if (v < idealLow) return (v - badLow) / (idealLow - badLow);
  return (badHigh - v) / (badHigh - idealHigh);
}

float clamp01(float v) {
  if (v < 0) return 0;
  if (v > 1) return 1;
  return v;
}

float median3(float a, float b, float c) {
  if (a > b) { float t = a; a = b; b = t; }
  if (b > c) { float t = b; b = c; c = t; }
  if (a > b) { float t = a; a = b; b = t; }
  return b;
}

// Is feather i on the side the threat is coming from?
bool onThreatSide(uint8_t i) {
  if (threatDir < 0) return i <= 1;
  if (threatDir > 0) return i >= 3;
  return true;  // straight ahead: the whole fan
}

void setFeather(uint8_t i, float target, float speed) {
  feathers[i].target = clamp01(target);
  feathers[i].speed = speed;
}

// ===================================================================
//  SENSORS: ENVIRONMENT
// ===================================================================
void readDht() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  // Keep the last good values if a read fails
  if (!isnan(t)) tempC = t;
  if (!isnan(h)) humidity = h;
  factorValid[F_TEMP] = !isnan(t);
  factorValid[F_HUM] = !isnan(h);
}

void readAnalogSensors(unsigned long now) {
  float light = analogRead(PIN_LIGHT) * 100.0 / 1023.0;
  float air = analogRead(PIN_AIR);

  // Smooth out the noisy analog readings
  if (!analogPrimed) {
    lightPct = light;
    airRaw = air;
    analogPrimed = true;
  } else {
    lightPct = lightPct * 0.85 + light * 0.15;
    airRaw = airRaw * 0.85 + air * 0.15;
  }
  factorValid[F_LIGHT] = true;
  factorValid[F_AIR] = (now - bootMs) > AIR_WARMUP_MS;
}

void updateScores() {
  factorScore[F_TEMP]  = rangeScore(tempC, TEMP_BAD_LOW, TEMP_IDEAL_LOW, TEMP_IDEAL_HI, TEMP_BAD_HIGH);
  factorScore[F_HUM]   = rangeScore(humidity, HUM_BAD_LOW, HUM_IDEAL_LOW, HUM_IDEAL_HI, HUM_BAD_HIGH);
  factorScore[F_LIGHT] = rangeScore(lightPct, LIGHT_BAD_LOW, LIGHT_IDEAL_LOW, LIGHT_IDEAL_HI, LIGHT_BAD_HIGH);
  factorScore[F_AIR]   = rangeScore(airRaw, AIR_BAD_LOW, AIR_IDEAL_LOW, AIR_IDEAL_HI, AIR_BAD_HIGH);

  // Overall = half the weighted average, half the worst factor.
  // One really bad factor (like cold) can ruin an egg on its own,
  // so it should drag the mood down even if the rest are perfect.
  float weighted = 0, weightSum = 0, worst = 1.0;
  for (uint8_t f = 0; f < NUM_FACTORS; f++) {
    if (!factorValid[f]) continue;   // ignore sensors that aren't ready or have failed
    weighted += factorScore[f] * FACTOR_WEIGHT[f];
    weightSum += FACTOR_WEIGHT[f];
    if (factorScore[f] < worst) worst = factorScore[f];
  }
  float target = (weightSum > 0) ? 0.5 * (weighted / weightSum) + 0.5 * worst : 0.5;
  overallScore = overallScore * 0.7 + target * 0.3;

  if (overallScore >= MOOD_CONTENT_MIN)      mood = MOOD_CONTENT;
  else if (overallScore >= MOOD_UNEASY_MIN)  mood = MOOD_UNEASY;
  else                                       mood = MOOD_DISTRESSED;
}

// ===================================================================
//  SENSORS: THREATS
// ===================================================================
float pingDistanceCm() {
  digitalWrite(PIN_US_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_US_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_US_TRIG, LOW);
  unsigned long echoUs = pulseIn(PIN_US_ECHO, HIGH, 25000UL);  // ~4 m max
  if (echoUs == 0) return -1;
  return echoUs / 58.0;
}

void readSonar() {
  sonarHistory[sonarIndex] = pingDistanceCm();
  sonarIndex = (sonarIndex + 1) % 3;

  // Median of the last 3 pings rejects single bad echoes
  float d = median3(sonarHistory[0], sonarHistory[1], sonarHistory[2]);
  if (d > ALERT_DIST_CM * 1.5) d = -1;  // too far away to matter

  // Approach speed, smoothed
  if (d > 0 && distanceCm > 0) {
    float raw = (distanceCm - d) / (SONAR_INTERVAL_MS / 1000.0);
    approachSpeed = approachSpeed * 0.7 + raw * 0.3;
  } else {
    approachSpeed *= 0.7;
  }
  distanceCm = d;
}

void readPirs(unsigned long now) {
  if (now - bootMs < PIR_WARMUP_MS) {
    pirLeft = pirCenter = pirRight = false;
    return;
  }
  pirLeft = digitalRead(PIN_PIR_LEFT) == HIGH;
  pirCenter = digitalRead(PIN_PIR_CENTER) == HIGH;
  pirRight = digitalRead(PIN_PIR_RIGHT) == HIGH;
}

void updateThreat(unsigned long now) {
  bool anyMotion = pirLeft || pirCenter || pirRight;
  bool seen = distanceCm > 0 && distanceCm < ALERT_DIST_CM;

  if (anyMotion) {
    if (motionStartMs == 0) motionStartMs = now;
  } else {
    motionStartMs = 0;
  }

  // How serious is it right now?
  ThreatLevel desired = THREAT_NONE;
  if (anyMotion || seen) desired = THREAT_ALERT;
  if (seen && distanceCm < WARN_DIST_CM) desired = THREAT_WARN;
  if (seen && approachSpeed > APPROACH_FAST_CM_S) desired = THREAT_WARN;
  if (motionStartMs && now - motionStartMs > MOTION_PERSIST_MS && desired < THREAT_WARN) desired = THREAT_WARN;
  if (seen && distanceCm < DEFEND_DIST_CM) desired = THREAT_DEFEND;

  // Where is it coming from? The sonar points straight ahead, so a
  // close echo means center; otherwise trust the side PIRs.
  if (seen && distanceCm < WARN_DIST_CM)   threatDir = 0;
  else if (pirLeft && !pirRight)           threatDir = -1;
  else if (pirRight && !pirLeft)           threatDir = 1;
  else if (anyMotion || seen)              threatDir = 0;

  // React at once, but calm down one step at a time
  if (desired >= threat) {
    if (desired > threat) {
      Serial.print(F(">> Threat: "));
      Serial.print(THREAT_NAME[desired]);
      Serial.print(F(" from the "));
      Serial.println(threatDir < 0 ? F("LEFT") : threatDir > 0 ? F("RIGHT") : F("FRONT"));
    }
    if (desired != THREAT_NONE || threat != THREAT_NONE) threatChangedMs = now;
    threat = desired;
  } else if (now - threatChangedMs > THREAT_HOLD_MS) {
    threat = (ThreatLevel)(threat - 1);
    threatChangedMs = now;
    Serial.print(F("<< Calming down to: "));
    Serial.println(THREAT_NAME[threat]);
  }
}

// ===================================================================
//  FEATHER ANIMATION
// ===================================================================
void animateMood(unsigned long now) {
  float t = now / 1000.0;

  // Uneasy: now and then, the worst feather twitches
  if (mood == MOOD_UNEASY && now >= nextTwitchMs) {
    float worst = 2.0;
    for (uint8_t i = 0; i < NUM_FEATHERS; i++) {
      int8_t f = FEATHER_FACTOR[i];
      if (f >= 0 && factorValid[f] && factorScore[f] < worst) {
        worst = factorScore[f];
        twitchFeather = i;
      }
    }
    twitchUntilMs = now + 250;
    nextTwitchMs = now + random(2500, 6000);
  }

  for (uint8_t i = 0; i < NUM_FEATHERS; i++) {
    int8_t f = FEATHER_FACTOR[i];
    float score = (f < 0) ? overallScore : (factorValid[f] ? factorScore[f] : 0.5);
    float base = 0.2 + 0.8 * score;  // never fully collapse in mood mode
    float target = base;
    float speed = 0.6;

    switch (mood) {
      case MOOD_CONTENT:
        // Slow ripple travelling across the fan
        target = base - 0.06 + 0.06 * sin(t * 1.2 - i * 0.7);
        speed = 0.5;
        break;

      case MOOD_UNEASY:
        target = base - 0.05;
        if (i == twitchFeather && now < twitchUntilMs) {
          target = base - 0.3;
          speed = 4.0;
        }
        break;

      case MOOD_DISTRESSED:
        // Sagging, with a shiver for 1.5 s of every 5 s
        target = base * 0.7;
        if ((now % 5000) < 1500) {
          target += 0.04 * sin(t * 25.0 + i);
          speed = 3.0;
        }
        break;
    }
    setFeather(i, target, speed);
  }
}

void animateDefense(unsigned long now) {
  float t = now / 1000.0;

  for (uint8_t i = 0; i < NUM_FEATHERS; i++) {
    bool facing = onThreatSide(i);
    float target = 1.0;
    float speed = 3.0;

    switch (threat) {
      case THREAT_ALERT:
        // Raise up, lean toward the intruder, quiver
        target = (facing ? 1.0 : 0.75) - 0.03 + 0.03 * sin(t * 30.0 + i);
        speed = 3.0;
        break;

      case THREAT_WARN:
        // Full display with a rattle every 0.7 s
        target = ((now % 700) < 120) ? 0.8 : 1.0;
        speed = 6.0;
        break;

      case THREAT_DEFEND:
        if (facing) {
          // Swat! Neighbouring feathers are out of step so they whip in turn.
          bool down = ((now + i * 110UL) / 220UL) % 2;
          target = down ? 0.2 : 1.0;
          speed = 8.0;
        } else {
          target = 1.0;
          speed = 6.0;
        }
        break;

      default:
        break;
    }
    setFeather(i, target, speed);
  }
}

void moveFeathers(float dt) {
  for (uint8_t i = 0; i < NUM_FEATHERS; i++) {
    Feather& f = feathers[i];
    float step = f.speed * dt;
    float diff = f.target - f.pos;
    if (fabs(diff) <= step) f.pos = f.target;
    else f.pos += (diff > 0 ? step : -step);

    int angle = (int)(FEATHER_DOWN_ANGLE[i] + f.pos * (FEATHER_UP_ANGLE[i] - FEATHER_DOWN_ANGLE[i]) + 0.5);
    if (angle != f.lastAngle) {   // only send changes, reduces buzzing
      f.servo.write(angle);
      f.lastAngle = angle;
    }
  }
}

// ===================================================================
//  SOUND
// ===================================================================
void updateSound(unsigned long now) {
#if USE_BUZZER
  bool wantHiss = (threat == THREAT_DEFEND) ||
                  (threat == THREAT_WARN && (now % 700) < 300);
  if (wantHiss) {
    // A goose hisses; random fast-changing pitch sounds like a hiss
    if (!hissing || now - lastHissChangeMs > 15) {
      tone(PIN_BUZZER, random(1800, 4500));
      lastHissChangeMs = now;
      hissing = true;
    }
  } else if (hissing) {
    noTone(PIN_BUZZER);
    hissing = false;
  }
#endif
}

// ===================================================================
//  REPORT
// ===================================================================
void printReport(unsigned long now) {
  Serial.print(F("T="));
  if (isnan(tempC)) Serial.print(F("--")); else Serial.print(tempC, 1);
  Serial.print(F("C  RH="));
  if (isnan(humidity)) Serial.print(F("--")); else Serial.print(humidity, 0);
  Serial.print(F("%  light="));
  Serial.print(lightPct, 0);
  Serial.print(F("%  air="));
  Serial.print(airRaw, 0);
  if (!factorValid[F_AIR]) Serial.print(F("(warming)"));

  Serial.print(F("  | scores"));
  for (uint8_t f = 0; f < NUM_FACTORS; f++) {
    Serial.print(' ');
    Serial.print(FACTOR_NAME[f]);
    Serial.print('=');
    if (factorValid[f]) Serial.print(factorScore[f], 2); else Serial.print(F("--"));
  }
  Serial.print(F("  overall="));
  Serial.print(overallScore, 2);
  Serial.print(F("  mood="));
  Serial.print(MOOD_NAME[mood]);

  Serial.print(F("  | dist="));
  if (distanceCm > 0) { Serial.print(distanceCm, 0); Serial.print(F("cm")); }
  else Serial.print(F("--"));
  Serial.print(F(" pir="));
  Serial.print(pirLeft ? 'L' : '-');
  Serial.print(pirCenter ? 'C' : '-');
  Serial.print(pirRight ? 'R' : '-');
  if (now - bootMs < PIR_WARMUP_MS) Serial.print(F("(warming)"));
  Serial.print(F(" threat="));
  Serial.println(THREAT_NAME[threat]);
}

// ===================================================================
//  SETUP & LOOP
// ===================================================================
void wakeUpAnimation() {
  // Feathers rise one at a time from the center outward, like a
  // peacock opening its tail
  const uint8_t order[NUM_FEATHERS] = {2, 1, 3, 0, 4};
  for (uint8_t n = 0; n < NUM_FEATHERS; n++) {
    uint8_t i = order[n];
    for (int step = 0; step <= 20; step++) {
      float p = step / 20.0;
      feathers[i].servo.write(FEATHER_DOWN_ANGLE[i] + p * (FEATHER_UP_ANGLE[i] - FEATHER_DOWN_ANGLE[i]));
      delay(15);
    }
    feathers[i].pos = 1.0;
    feathers[i].target = 1.0;
    feathers[i].lastAngle = FEATHER_UP_ANGLE[i];
  }
  delay(400);
}

void setup() {
  Serial.begin(9600);
  Serial.println(F("Goose Nest Guardian waking up..."));

  pinMode(PIN_PIR_LEFT, INPUT);
  pinMode(PIN_PIR_CENTER, INPUT);
  pinMode(PIN_PIR_RIGHT, INPUT);
  pinMode(PIN_US_TRIG, OUTPUT);
  pinMode(PIN_US_ECHO, INPUT);
  digitalWrite(PIN_US_TRIG, LOW);
#if USE_BUZZER
  pinMode(PIN_BUZZER, OUTPUT);
#endif

  dht.begin();
  randomSeed(analogRead(A3));  // floating pin gives a random seed

  for (uint8_t i = 0; i < NUM_FEATHERS; i++) {
    feathers[i].servo.attach(PIN_FEATHER[i]);
    feathers[i].servo.write(FEATHER_DOWN_ANGLE[i]);
    feathers[i].pos = 0;
    feathers[i].target = 0;
    feathers[i].speed = 1.0;
    feathers[i].lastAngle = FEATHER_DOWN_ANGLE[i];
  }
  for (uint8_t f = 0; f < NUM_FACTORS; f++) {
    factorScore[f] = 0.5;
    factorValid[f] = false;
  }
  delay(500);
  wakeUpAnimation();

  bootMs = millis();
  lastAnimMs = bootMs;
  Serial.println(F("Awake. Motion sensors warming up for 30 s, air sensor for 60 s."));
}

void loop() {
  unsigned long now = millis();

  // --- Sense ---
  if (now - lastDhtMs >= DHT_INTERVAL_MS) {
    lastDhtMs = now;
    readDht();
  }
  if (now - lastAnalogMs >= ANALOG_INTERVAL_MS) {
    lastAnalogMs = now;
    readAnalogSensors(now);
    updateScores();
  }
  if (now - lastSonarMs >= SONAR_INTERVAL_MS) {
    lastSonarMs = now;
    readSonar();
  }
  readPirs(now);
  updateThreat(now);

  // --- Feel & act ---
  now = millis();  // the sonar ping may have taken a few ms
  if (now - lastAnimMs >= ANIM_INTERVAL_MS) {
    float dt = (now - lastAnimMs) / 1000.0;
    lastAnimMs = now;
    if (threat == THREAT_NONE) animateMood(now);
    else animateDefense(now);
    moveFeathers(dt);
  }
  updateSound(now);

  // --- Report ---
  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = now;
    printReport(now);
  }
}
