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
 *    4 = far right  -> quietness
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
 *    ALERT  -> motion seen, something within range, or a sudden loud
 *              noise: the fan rises, leans toward the intruder and
 *              quivers
 *    WARN   -> something is getting close or coming fast: the fan
 *              spreads fully, rattles and hisses (if a buzzer is fitted)
 *    DEFEND -> something is right at the nest: the feathers on the
 *              intruder's side swat at it hard and fast
 *
 *  ------------------------------------------------------------------
 *  HARDWARE (Arduino Uno / Nano / Mega)
 *  ------------------------------------------------------------------
 *    DHT11 temperature/humidity (S pin)  -> D2
 *    HC-SR501 PIR, LEFT  (OUT)           -> D4
 *    HC-SR501 PIR, RIGHT (OUT)           -> D5
 *    HC-SR04 ultrasonic LEFT  TRIG/ECHO  -> D6 / D7
 *    HC-SR04 ultrasonic RIGHT TRIG/ECHO  -> D8 / A3
 *    Servo feathers 0..4                 -> D3, D9, D10, D11, D12
 *    Photoresistor (LDR)                 -> A0  (LDR to 5V, 10k resistor
 *                                                to GND, junction to A0:
 *                                                brighter = higher)
 *    KY-037 sound sensor (AO pin)        -> A1
 *    Passive buzzer (optional)           -> A2
 *
 *  Aim the two ultrasonic sensors about 30 degrees outward, one to the
 *  left and one to the right, so their beams overlap a little in front.
 *  Point the two PIR sensors the same way.
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
const uint8_t PIN_DHT          = 2;
const uint8_t PIN_PIR_LEFT     = 4;
const uint8_t PIN_PIR_RIGHT    = 5;
const uint8_t PIN_US_LEFT_TRIG = 6;
const uint8_t PIN_US_LEFT_ECHO = 7;
const uint8_t PIN_US_RIGHT_TRIG = 8;
const uint8_t PIN_US_RIGHT_ECHO = A3;
const uint8_t PIN_LIGHT        = A0;
const uint8_t PIN_SOUND        = A1;
const uint8_t PIN_BUZZER       = A2;
const uint8_t PIN_RANDOM_SEED  = A5;   // leave unconnected

const uint8_t NUM_FEATHERS = 5;
const uint8_t PIN_FEATHER[NUM_FEATHERS] = {3, 9, 10, 11, 12};

#define DHT_TYPE DHT11
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

// Temperature (deg C): goose eggs like about 37.5 C (99.5 F).
// The DHT11 reads in whole degrees and is only accurate to +/-2 C,
// so the ideal window is a little wider than an incubator would use.
const float TEMP_BAD_LOW   = 35.0;
const float TEMP_IDEAL_LOW = 37.0;
const float TEMP_IDEAL_HI  = 38.0;
const float TEMP_BAD_HIGH  = 40.0;

// Relative humidity (%): around 50-55 % normally, 70-80 % at lockdown.
// The DHT11 only measures 20-80 % (+/-5 %).
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

// Noise (KY-037 peak-to-peak reading, 0-1023, higher = louder).
// Nesting geese pick quiet spots. These numbers depend on your
// sensor: watch "noise=" in the Serial Monitor in a quiet room and
// while talking nearby, then set NOISE_IDEAL_HI between the two.
const float NOISE_BAD_LOW   = -1.0;
const float NOISE_IDEAL_LOW =  0.0;
const float NOISE_IDEAL_HI  = 15.0;
const float NOISE_BAD_HIGH  = 80.0;

// How much each factor counts toward the overall comfort score
enum Factor { F_TEMP, F_HUM, F_LIGHT, F_NOISE, NUM_FACTORS };
const float FACTOR_WEIGHT[NUM_FACTORS] = {0.45, 0.30, 0.10, 0.15};
const char* const FACTOR_NAME[NUM_FACTORS] = {"temp", "humidity", "light", "quiet"};

// Which feather shows which factor (-1 = the overall score)
const int8_t FEATHER_FACTOR[NUM_FEATHERS] = {F_TEMP, F_HUM, -1, F_LIGHT, F_NOISE};

// Mood thresholds on the overall score (0..1)
const float MOOD_CONTENT_MIN = 0.75;
const float MOOD_UNEASY_MIN  = 0.45;

// ===================================================================
//  DEFENSE SETTINGS
// ===================================================================
const float ALERT_DIST_CM       = 200.0;  // notice things inside this range
const float WARN_DIST_CM        = 100.0;  // threat display inside this range
const float DEFEND_DIST_CM      =  40.0;  // swat at things inside this range
const float CENTER_DIFF_CM      =  25.0;  // both sonars within this of each other = straight ahead
const float APPROACH_FAST_CM_S  =  40.0;  // closing faster than this = WARN
const unsigned long MOTION_PERSIST_MS = 4000;  // lingering motion -> WARN
const unsigned long THREAT_HOLD_MS    = 4000;  // calm down one level after this

// A sudden sound this far above the background level startles the
// nest into ALERT. Raise it if it jumps at ordinary noises.
const float LOUD_MARGIN   = 60.0;
const unsigned long LOUD_STARTLE_MS = 1500;
// Ignore the microphone this long after defending, so the nest's own
// servos and hissing don't keep it on alert forever.
const unsigned long SOUND_IGNORE_AFTER_THREAT_MS = 1500;

// ===================================================================
//  TIMING
// ===================================================================
const unsigned long PIR_WARMUP_MS      = 60000;  // HC-SR501 gives false hits for ~1 min at power-up
const unsigned long DHT_INTERVAL_MS    = 2000;   // DHT11 can't be read faster than once a second
const unsigned long LIGHT_INTERVAL_MS  = 200;
const unsigned long SOUND_INTERVAL_MS  = 100;
const unsigned long SOUND_WINDOW_US    = 20000;  // listen for 20 ms each time
const unsigned long SONAR_INTERVAL_MS  = 50;     // sonars take turns, so each pings every 100 ms
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

struct Sonar {
  uint8_t trigPin;
  uint8_t echoPin;
  float history[3];        // last 3 raw pings, for a median filter
  uint8_t historyIndex;
  float distanceCm;        // filtered distance, -1 = nothing in range
  float approachSpeed;     // cm/s, positive = getting closer
  unsigned long lastPingMs;
};
enum { SONAR_LEFT, SONAR_RIGHT, NUM_SONARS };
Sonar sonars[NUM_SONARS] = {
  {PIN_US_LEFT_TRIG,  PIN_US_LEFT_ECHO,  {-1, -1, -1}, 0, -1, 0, 0},
  {PIN_US_RIGHT_TRIG, PIN_US_RIGHT_ECHO, {-1, -1, -1}, 0, -1, 0, 0},
};
uint8_t nextSonar = 0;

enum Mood { MOOD_CONTENT, MOOD_UNEASY, MOOD_DISTRESSED };
const char* const MOOD_NAME[] = {"CONTENT", "UNEASY", "DISTRESSED"};

enum ThreatLevel { THREAT_NONE, THREAT_ALERT, THREAT_WARN, THREAT_DEFEND };
const char* const THREAT_NAME[] = {"none", "ALERT", "WARN", "DEFEND"};

// Environment readings
float tempC = NAN;
float humidity = NAN;
float lightPct = 0;
bool  lightPrimed = false;
float soundLevel = 0;      // latest peak-to-peak loudness
float soundAvg = 0;        // slow background loudness
bool  soundPrimed = false;

float factorScore[NUM_FACTORS];
bool  factorValid[NUM_FACTORS];
float overallScore = 0.5;
Mood  mood = MOOD_UNEASY;

// Threat sensing
bool pirLeft = false, pirRight = false;
unsigned long motionStartMs = 0;
unsigned long loudUntilMs = 0;
unsigned long lastThreatMs = 0;

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
unsigned long lastDhtMs = 0, lastLightMs = 0, lastSoundMs = 0, lastSonarMs = 0;
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

void readLight() {
  float light = analogRead(PIN_LIGHT) * 100.0 / 1023.0;
  // Smooth out flicker and passing shadows
  lightPct = lightPrimed ? lightPct * 0.85 + light * 0.15 : light;
  lightPrimed = true;
  factorValid[F_LIGHT] = true;
}

void readSound(unsigned long now) {
  // The KY-037's AO pin gives the raw microphone wave around a middle
  // value. Loudness = how far it swings (peak-to-peak) in a short window.
  int lo = 1023, hi = 0;
  unsigned long start = micros();
  while (micros() - start < SOUND_WINDOW_US) {
    int v = analogRead(PIN_SOUND);
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  soundLevel = hi - lo;

  // While defending (and just after), the nest's own servos and
  // hissing are the loudest thing around, so don't listen.
  bool selfNoise = threat != THREAT_NONE ||
                   (lastThreatMs && now - lastThreatMs < SOUND_IGNORE_AFTER_THREAT_MS);
  if (selfNoise) return;

  if (!soundPrimed) {
    soundAvg = soundLevel;
    soundPrimed = true;
  }
  // A sudden noise well above the background startles the nest
  if (soundLevel > soundAvg + LOUD_MARGIN) loudUntilMs = now + LOUD_STARTLE_MS;

  // Background loudness, averaged over about 5 seconds
  soundAvg = soundAvg * 0.98 + soundLevel * 0.02;
  factorValid[F_NOISE] = true;
}

void updateScores() {
  factorScore[F_TEMP]  = rangeScore(tempC, TEMP_BAD_LOW, TEMP_IDEAL_LOW, TEMP_IDEAL_HI, TEMP_BAD_HIGH);
  factorScore[F_HUM]   = rangeScore(humidity, HUM_BAD_LOW, HUM_IDEAL_LOW, HUM_IDEAL_HI, HUM_BAD_HIGH);
  factorScore[F_LIGHT] = rangeScore(lightPct, LIGHT_BAD_LOW, LIGHT_IDEAL_LOW, LIGHT_IDEAL_HI, LIGHT_BAD_HIGH);
  factorScore[F_NOISE] = rangeScore(soundAvg, NOISE_BAD_LOW, NOISE_IDEAL_LOW, NOISE_IDEAL_HI, NOISE_BAD_HIGH);

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
float pingDistanceCm(const Sonar& s) {
  digitalWrite(s.trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(s.trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(s.trigPin, LOW);
  unsigned long echoUs = pulseIn(s.echoPin, HIGH, 25000UL);  // ~4 m max
  if (echoUs == 0) return -1;
  return echoUs / 58.0;
}

// Ping one sonar per call. Taking turns stops one sensor from
// hearing the other's echo.
void readNextSonar(unsigned long now) {
  Sonar& s = sonars[nextSonar];
  nextSonar = (nextSonar + 1) % NUM_SONARS;

  s.history[s.historyIndex] = pingDistanceCm(s);
  s.historyIndex = (s.historyIndex + 1) % 3;

  // Median of the last 3 pings rejects single bad echoes
  float d = median3(s.history[0], s.history[1], s.history[2]);
  if (d > ALERT_DIST_CM * 1.5) d = -1;  // too far away to matter

  // Approach speed, smoothed
  float dt = (now - s.lastPingMs) / 1000.0;
  if (d > 0 && s.distanceCm > 0 && dt > 0) {
    float raw = (s.distanceCm - d) / dt;
    s.approachSpeed = s.approachSpeed * 0.7 + raw * 0.3;
  } else {
    s.approachSpeed *= 0.7;
  }
  s.distanceCm = d;
  s.lastPingMs = now;
}

void readPirs(unsigned long now) {
  if (now - bootMs < PIR_WARMUP_MS) {
    pirLeft = pirRight = false;
    return;
  }
  pirLeft = digitalRead(PIN_PIR_LEFT) == HIGH;
  pirRight = digitalRead(PIN_PIR_RIGHT) == HIGH;
}

void updateThreat(unsigned long now) {
  const Sonar& L = sonars[SONAR_LEFT];
  const Sonar& R = sonars[SONAR_RIGHT];
  bool seenL = L.distanceCm > 0 && L.distanceCm < ALERT_DIST_CM;
  bool seenR = R.distanceCm > 0 && R.distanceCm < ALERT_DIST_CM;
  bool seen = seenL || seenR;

  // Closest thing either sonar can see
  float nearest = -1;
  float approach = 0;
  if (seenL) { nearest = L.distanceCm; approach = L.approachSpeed; }
  if (seenR && (nearest < 0 || R.distanceCm < nearest)) { nearest = R.distanceCm; approach = R.approachSpeed; }

  bool anyMotion = pirLeft || pirRight;
  bool startled = now < loudUntilMs;

  if (anyMotion) {
    if (motionStartMs == 0) motionStartMs = now;
  } else {
    motionStartMs = 0;
  }

  // How serious is it right now?
  ThreatLevel desired = THREAT_NONE;
  if (anyMotion || seen || startled) desired = THREAT_ALERT;
  if (seen && nearest < WARN_DIST_CM) desired = THREAT_WARN;
  if (seen && approach > APPROACH_FAST_CM_S) desired = THREAT_WARN;
  if (motionStartMs && now - motionStartMs > MOTION_PERSIST_MS && desired < THREAT_WARN) desired = THREAT_WARN;
  if (seen && nearest < DEFEND_DIST_CM) desired = THREAT_DEFEND;

  // Where is it coming from? The sonars give the best direction;
  // the PIRs are used when nothing is close enough to measure.
  if (seenL && seenR && fabs(L.distanceCm - R.distanceCm) < CENTER_DIFF_CM) threatDir = 0;
  else if (seenL && (!seenR || L.distanceCm < R.distanceCm))               threatDir = -1;
  else if (seenR)                                                          threatDir = 1;
  else if (pirLeft && !pirRight)                                           threatDir = -1;
  else if (pirRight && !pirLeft)                                           threatDir = 1;
  else if (anyMotion || startled)                                          threatDir = 0;

  // React at once, but calm down one step at a time
  if (desired >= threat) {
    if (desired > threat) {
      Serial.print(F(">> Threat: "));
      Serial.print(THREAT_NAME[desired]);
      Serial.print(F(" from the "));
      Serial.print(threatDir < 0 ? F("LEFT") : threatDir > 0 ? F("RIGHT") : F("FRONT"));
      if (startled && !anyMotion && !seen) Serial.print(F(" (loud noise)"));
      Serial.println();
    }
    if (desired != THREAT_NONE || threat != THREAT_NONE) threatChangedMs = now;
    threat = desired;
  } else if (now - threatChangedMs > THREAT_HOLD_MS) {
    threat = (ThreatLevel)(threat - 1);
    threatChangedMs = now;
    Serial.print(F("<< Calming down to: "));
    Serial.println(THREAT_NAME[threat]);
  }
  if (threat != THREAT_NONE) lastThreatMs = now;
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
//  SOUND OUTPUT
// ===================================================================
void updateBuzzer(unsigned long now) {
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
void printSonar(const Sonar& s) {
  if (s.distanceCm > 0) { Serial.print(s.distanceCm, 0); Serial.print(F("cm")); }
  else Serial.print(F("--"));
}

void printReport(unsigned long now) {
  Serial.print(F("T="));
  if (isnan(tempC)) Serial.print(F("--")); else Serial.print(tempC, 0);
  Serial.print(F("C  RH="));
  if (isnan(humidity)) Serial.print(F("--")); else Serial.print(humidity, 0);
  Serial.print(F("%  light="));
  Serial.print(lightPct, 0);
  Serial.print(F("%  noise="));
  Serial.print(soundAvg, 0);
  Serial.print(F(" (now "));
  Serial.print(soundLevel, 0);
  Serial.print(')');

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

  Serial.print(F("  | sonar L="));
  printSonar(sonars[SONAR_LEFT]);
  Serial.print(F(" R="));
  printSonar(sonars[SONAR_RIGHT]);
  Serial.print(F(" pir="));
  Serial.print(pirLeft ? 'L' : '-');
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
  pinMode(PIN_PIR_RIGHT, INPUT);
  for (uint8_t s = 0; s < NUM_SONARS; s++) {
    pinMode(sonars[s].trigPin, OUTPUT);
    pinMode(sonars[s].echoPin, INPUT);
    digitalWrite(sonars[s].trigPin, LOW);
  }
#if USE_BUZZER
  pinMode(PIN_BUZZER, OUTPUT);
#endif

  dht.begin();
  randomSeed(analogRead(PIN_RANDOM_SEED));

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
  Serial.println(F("Awake. Motion sensors warming up for 60 s."));
}

void loop() {
  unsigned long now = millis();

  // --- Sense ---
  if (now - lastDhtMs >= DHT_INTERVAL_MS) {
    lastDhtMs = now;
    readDht();
  }
  if (now - lastLightMs >= LIGHT_INTERVAL_MS) {
    lastLightMs = now;
    readLight();
    updateScores();
  }
  if (now - lastSoundMs >= SOUND_INTERVAL_MS) {
    lastSoundMs = now;
    readSound(now);
  }
  if (now - lastSonarMs >= SONAR_INTERVAL_MS) {
    lastSonarMs = now;
    readNextSonar(now);
  }
  readPirs(now);
  updateThreat(now);

  // --- Feel & act ---
  now = millis();  // sonar and sound reads take a few ms
  if (now - lastAnimMs >= ANIM_INTERVAL_MS) {
    float dt = (now - lastAnimMs) / 1000.0;
    lastAnimMs = now;
    if (threat == THREAT_NONE) animateMood(now);
    else animateDefense(now);
    moveFeathers(dt);
  }
  updateBuzzer(now);

  // --- Report ---
  if (now - lastReportMs >= REPORT_INTERVAL_MS) {
    lastReportMs = now;
    printReport(now);
  }
}
