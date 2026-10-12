/*
 * NEST RADIAL
 *
 * Draws five minutes of readings from the goose nest as one radial record.
 * Time runs clockwise from the open wedge at the top; one full turn is
 * five minutes. Each sensor has its own ring, inner to outer:
 *
 *   proximity    hairlines reach in toward the centre when something comes close
 *   temperature  one continuous black line that swells when it is warmer
 *   humidity     translucent dots that pile up into darker texture
 *   light        radial hairlines, longer when it is brighter
 *   sound        tiny dots when quiet; a loud sound fires a red dot on a stem
 *
 * After five minutes it stops and saves a high-resolution PNG and a vector
 * PDF into the "exports" folder next to this sketch.
 *
 * Keys:  R = restart   S = save now   D = switch demo mode on/off
 *
 * The Arduino must run nest_feathers.ino with SEND_TO_PROCESSING = true,
 * and the Arduino IDE's Serial Monitor must be closed.
 */

import processing.serial.*;
import processing.pdf.*;

// ---------------- Settings ----------------
String  PORT_NAME = "";         // e.g. "COM3" or "/dev/cu.usbmodem14101"; "" = last port in the list
int     BAUD = 9600;
boolean DEMO = false;           // true = invent readings, no Arduino needed
float   DURATION = 300;         // seconds for one full turn (300 = 5 minutes)
float   DISPLAY_SCALE = 0.6;    // window size: 1.0 = 1400 x 1600 pixels
int     EXPORT_SCALE = 3;       // PNG size: 3 = 4200 x 4800 pixels

// Sensor ranges, the same as in the Arduino sketch
float TEMP_LOW  = 5,   TEMP_HIGH  = 25;     // deg C
float HUM_LOW   = 35,  HUM_HIGH   = 80;     // %
float LIGHT_LOW = 450, LIGHT_HIGH = 1000;   // raw reading
float SOUND_LOW = 20,  SOUND_HIGH = 200;    // loudness
float NEAR_FAR  = 70,  NEAR_CLOSE = 5;      // cm

// Colours
color PAPER = color(247, 245, 240);
color INK   = color(17, 17, 17);
color RED   = color(193, 18, 31);

// Layout, in drawing units (the full drawing is 1400 x 1600)
float W = 1400, H = 1600;
float CX = 700, CY = 720;                   // centre of the circle
float R_NEAR_IN = 110, R_NEAR_OUT = 300;
float R_TEMP = 150;
float R_HUM_IN = 190, R_HUM_OUT = 285;
float R_LIGHT_IN = 320, R_LIGHT_OUT = 415;
float R_SOUND = 445, R_SOUND_MAX = 640;
float R_GRID = 665;
float GAP = radians(16);                    // open wedge at the top

// ---------------- State ----------------
class Sample {
  float t, dist, sound, temp, hum, light;
  float j1, j2;   // fixed random jitter, so the export looks the same as the screen
  float[] hj = new float[6];   // jitter for the humidity dots

  Sample(float t, float dist, float sound, float temp, float hum, float light) {
    this.t = t;
    this.dist = dist;
    this.sound = sound;
    this.temp = temp;
    this.hum = hum;
    this.light = light;
    j1 = random(-1, 1);
    j2 = random(-1, 1);
    for (int i = 0; i < hj.length; i++) hj[i] = random(-1, 1);
  }
}

ArrayList<Sample> samples = new ArrayList<Sample>();    // everything drawn so far
ArrayList<Sample> incoming = new ArrayList<Sample>();   // new readings from the Arduino
Serial port;
PGraphics grid;     // paper, grid and title
PGraphics marks;    // the data marks, added to as readings arrive
PFont font;
int startMillis;
boolean finished = false;
String dateText = "";
String savedName = "";
String portText = "";
Sample latest;
float demoTime = 0;
float demoNearStart = -100;

// ---------------- Setup ----------------
void settings() {
  size(round(W * DISPLAY_SCALE), round(H * DISPLAY_SCALE));
  smooth(8);
}

void setup() {
  font = createFont("Serif", 32);
  grid = createGraphics(width, height);
  marks = createGraphics(width, height);
  if (!DEMO) connect();
  restart();
}

void connect() {
  String[] ports = Serial.list();
  println("Serial ports:");
  printArray(ports);
  if (ports.length == 0) {
    println("No serial ports found - switching to demo mode");
    DEMO = true;
    return;
  }
  String name = PORT_NAME.equals("") ? ports[ports.length - 1] : PORT_NAME;
  try {
    port = new Serial(this, name, BAUD);
    port.bufferUntil('\n');
    portText = name;
    println("Reading from " + name);
  }
  catch (Exception e) {
    println("Could not open " + name + " - switching to demo mode. Is the Serial Monitor still open?");
    DEMO = true;
  }
}

void restart() {
  synchronized (this) {
    incoming.clear();
  }
  samples.clear();
  latest = null;
  finished = false;
  savedName = "";
  startMillis = millis();
  demoTime = 0;
  demoNearStart = -100;
  noiseSeed((int) random(100000));
  dateText = nf(day(), 2) + "." + nf(month(), 2) + "." + year() + "   " + nf(hour(), 2) + ":" + nf(minute(), 2);

  grid.beginDraw();
  grid.pushMatrix();
  grid.scale(DISPLAY_SCALE);
  drawPaperAndGrid(grid);
  grid.popMatrix();
  grid.endDraw();

  marks.beginDraw();
  marks.clear();
  marks.endDraw();
}

// ---------------- Reading data ----------------
float elapsed() {
  return (millis() - startMillis) / 1000.0;
}

// Called by the Serial library each time a full line arrives
void serialEvent(Serial p) {
  String line = p.readStringUntil('\n');
  if (line == null || DEMO) return;
  String[] v = split(trim(line), ',');
  if (v.length != 5) return;             // not a data line
  float[] n = new float[5];
  for (int i = 0; i < 5; i++) {
    n[i] = float(v[i]);
    if (Float.isNaN(n[i])) return;       // not a number
  }
  addReading(elapsed(), n[0], n[1], n[2], n[3], n[4]);
}

synchronized void addReading(float t, float dist, float sound, float temp, float hum, float light) {
  if (finished || t > DURATION) return;
  incoming.add(new Sample(t, dist, sound, temp, hum, light));
}

// Invented readings, about 8 per second, for trying the look without the Arduino
void makeDemoReadings() {
  float now = elapsed();
  while (demoTime < now) {
    demoTime += 0.125;
    float t = demoTime;
    float temp = 13 + 9 * noise(t * 0.01) + 1.5 * sin(t * 0.04);
    float hum = 38 + 36 * noise(100 + t * 0.02);
    float light = 560 + 440 * noise(200 + t * 0.03);
    if (noise(300 + t * 0.15) > 0.68) light -= 300;          // a passing shadow
    float sound = 8 + 10 * noise(400 + t * 2);
    if (random(1) < 0.01) sound = random(50, 280);          // a clap or a call
    if (t > demoNearStart + 7 && random(1) < 0.005) demoNearStart = t;
    float dist = -1;
    float k = t - demoNearStart;
    if (k >= 0 && k < 7) dist = 70 - 62 * sin(PI * k / 7);  // something comes close, then leaves
    addReading(t, dist, sound, temp, hum, light);
  }
}

// ---------------- Drawing ----------------
void draw() {
  if (DEMO && !finished) makeDemoReadings();

  // Move new readings onto the marks layer
  ArrayList<Sample> fresh;
  synchronized (this) {
    fresh = new ArrayList<Sample>(incoming);
    incoming.clear();
  }
  if (fresh.size() > 0) {
    marks.beginDraw();
    marks.pushMatrix();
    marks.scale(DISPLAY_SCALE);
    for (Sample s : fresh) {
      Sample prev = samples.size() > 0 ? samples.get(samples.size() - 1) : null;
      drawSample(marks, prev, s);
      samples.add(s);
      latest = s;
    }
    marks.popMatrix();
    marks.endDraw();
  }

  if (!finished && elapsed() >= DURATION) {
    finished = true;
    saveRecord();
  }

  image(grid, 0, 0);
  image(marks, 0, 0);

  pushMatrix();
  scale(DISPLAY_SCALE);
  if (!finished) drawCursor(g, elapsed());
  drawReadout(g);
  popMatrix();
}

// Angle around the circle for a time in seconds
float angleAt(float t) {
  float f = constrain(t / DURATION, 0, 1);
  return -HALF_PI + GAP / 2 + f * (TWO_PI - GAP);
}

// 0..1 for where a value sits between low and high (works when low > high too)
float norm01(float v, float low, float high) {
  return constrain((v - low) / (high - low), 0, 1);
}

// Draw one reading. prev is the reading before it (or null), for the temperature line.
void drawSample(PGraphics g, Sample prev, Sample s) {
  float a = angleAt(s.t);
  g.pushMatrix();
  g.translate(CX, CY);

  // 1. Temperature: one continuous black line that swells when warmer
  if (prev != null && s.t - prev.t < 3) {
    float tn = norm01(s.temp, TEMP_LOW, TEMP_HIGH);
    float pa = angleAt(prev.t);
    g.noFill();
    g.stroke(INK);
    g.strokeCap(ROUND);
    g.strokeWeight(0.5 + tn * 11);
    g.line(cos(pa) * R_TEMP, sin(pa) * R_TEMP, cos(a) * R_TEMP, sin(a) * R_TEMP);
  }

  // 2. Humidity: a scatter of translucent dots. Wetter air fills the band
  //    further out with bigger dots, so it builds up darker.
  float hn = norm01(s.hum, HUM_LOW, HUM_HIGH);
  float top = lerp(R_HUM_IN + 10, R_HUM_OUT, hn);
  g.noStroke();
  g.fill(INK, 26);
  for (int i = 0; i < 3; i++) {
    float hr = lerp(R_HUM_IN, top, (s.hj[i] + 1) / 2);
    float ha = a + s.hj[i + 3] * radians(0.8);
    float hd = 1.5 + hn * 6 * abs(s.hj[i + 3]);
    g.ellipse(cos(ha) * hr, sin(ha) * hr, hd, hd);
  }
  if (s.j1 > 0.6) {
    g.stroke(INK, 22);
    g.strokeWeight(0.3);
    g.line(cos(a) * R_TEMP, sin(a) * R_TEMP, cos(a) * top, sin(a) * top);
  }

  // 3. Light: radial hairlines, longer when brighter
  float ln = norm01(s.light, LIGHT_LOW, LIGHT_HIGH);
  float la = a + s.j1 * radians(0.3);
  float lr = R_LIGHT_IN + 4 + ln * (R_LIGHT_OUT - R_LIGHT_IN - 4);
  g.stroke(INK, 70);
  g.strokeWeight(0.4);
  g.line(cos(la) * R_LIGHT_IN, sin(la) * R_LIGHT_IN, cos(la) * lr, sin(la) * lr);

  // 4. Sound: tiny dots when quiet, a red dot on a stem when loud
  float sn = norm01(s.sound, SOUND_LOW, SOUND_HIGH);
  if (sn <= 0.02) {
    float sr = R_SOUND + s.j1 * 4;
    g.noStroke();
    g.fill(INK, 90);
    g.ellipse(cos(a) * sr, sin(a) * sr, 1.6, 1.6);
  } else {
    float tip = R_SOUND + 10 + sn * (R_SOUND_MAX - R_SOUND - 10);
    g.stroke(RED, 150);
    g.strokeWeight(0.45);
    g.line(cos(a) * R_SOUND, sin(a) * R_SOUND, cos(a) * tip, sin(a) * tip);
    float sd = 3 + sn * 9;
    g.noStroke();
    g.fill(RED, 210);
    g.ellipse(cos(a) * tip, sin(a) * tip, sd, sd);
    if (sn > 0.5) {
      g.noFill();
      g.stroke(RED, 120);
      g.strokeWeight(0.5);
      g.ellipse(cos(a) * tip, sin(a) * tip, sd * 2.2, sd * 2.2);
    }
  }

  // 5. Proximity: hairlines reach in toward the centre, deeper when closer
  if (s.dist > 0 && s.dist <= NEAR_FAR) {
    float nn = norm01(s.dist, NEAR_FAR, NEAR_CLOSE);
    float rin = lerp(R_NEAR_OUT, R_NEAR_IN, nn);
    float na = a + s.j2 * radians(1.5);
    g.stroke(INK, 110);
    g.strokeWeight(0.35);
    g.line(cos(a) * R_NEAR_OUT, sin(a) * R_NEAR_OUT, cos(na) * rin, sin(na) * rin);
    g.noFill();
    g.stroke(INK, 160);
    g.strokeWeight(0.5);
    float nd = 4 + nn * 8;
    g.ellipse(cos(na) * rin, sin(na) * rin, nd, nd);
  }

  g.popMatrix();
}

// Paper, the polar grid, ring labels and the title block
void drawPaperAndGrid(PGraphics g) {
  g.background(PAPER);
  g.textFont(font);
  g.pushMatrix();
  g.translate(CX, CY);

  float start = angleAt(0);
  float end = angleAt(DURATION);

  // Faint spokes every minute
  g.stroke(INK, 18);
  g.strokeWeight(0.5);
  for (int m = 0; m <= DURATION / 60; m++) {
    float a = angleAt(m * 60);
    g.line(cos(a) * R_NEAR_IN, sin(a) * R_NEAR_IN, cos(a) * R_GRID, sin(a) * R_GRID);
  }

  // Faint rings at the band edges, left open at the gap
  g.noFill();
  g.stroke(INK, 35);
  float[] rings = { R_NEAR_IN, R_TEMP, R_HUM_IN, R_HUM_OUT, R_LIGHT_IN, R_LIGHT_OUT, R_SOUND };
  for (float r : rings) {
    g.arc(0, 0, r * 2, r * 2, start, end);
  }
  g.stroke(INK, 120);
  g.strokeWeight(0.7);
  g.arc(0, 0, R_GRID * 2, R_GRID * 2, start, end);

  // Ticks every 10 seconds, labels every minute
  g.textAlign(CENTER, CENTER);
  g.textSize(13);
  for (int sec = 0; sec <= DURATION; sec += 10) {
    float a = angleAt(sec);
    boolean major = sec % 60 == 0;
    float len = major ? 14 : 6;
    g.stroke(INK, major ? 150 : 70);
    g.strokeWeight(major ? 0.8 : 0.5);
    g.line(cos(a) * R_GRID, sin(a) * R_GRID, cos(a) * (R_GRID + len), sin(a) * (R_GRID + len));
    if (major) {
      g.fill(INK, 170);
      g.text(sec / 60 + "'", cos(a) * (R_GRID + 30), sin(a) * (R_GRID + 30));
    }
  }

  // Ring names, set in the open wedge
  g.textSize(9);
  g.fill(INK, 150);
  g.text("NEAR", 0, -R_NEAR_IN);
  g.text("TEMP", 0, -R_TEMP);
  g.text("HUMIDITY", 0, -(R_HUM_IN + R_HUM_OUT) / 2);
  g.text("LIGHT", 0, -(R_LIGHT_IN + R_LIGHT_OUT) / 2);
  g.text("SOUND", 0, -R_SOUND);
  g.popMatrix();

  // Title block
  g.textAlign(CENTER, BASELINE);
  g.fill(INK);
  g.textSize(28);
  g.text("Nest Record", CX, 1470);
  g.fill(INK, 170);
  g.textSize(14);
  g.text(dateText + "   ·   " + int(DURATION / 60) + " minutes   ·   Canada goose nest", CX, 1500);
  g.textSize(10);
  g.text("inner to outer:  proximity  ·  temperature  ·  humidity  ·  light  ·  sound", CX, 1526);
}

// The red line showing "now" (screen only)
void drawCursor(PGraphics g, float t) {
  float a = angleAt(t);
  g.stroke(RED, 160);
  g.strokeWeight(0.8);
  g.line(CX + cos(a) * R_NEAR_IN, CY + sin(a) * R_NEAR_IN, CX + cos(a) * R_GRID, CY + sin(a) * R_GRID);
}

// Live numbers in the top-left corner (screen only, not in the exports)
void drawReadout(PGraphics g) {
  g.textFont(font);
  g.textAlign(LEFT, BASELINE);
  g.textSize(13);
  g.fill(INK, 160);
  float t = min(elapsed(), DURATION);
  String mode = DEMO ? "DEMO" : "LIVE " + portText;
  g.text(mode + "   " + nf(int(t) / 60, 1) + ":" + nf(int(t) % 60, 2) + " / " + nf(int(DURATION) / 60, 1) + ":" + nf(int(DURATION) % 60, 2), 40, 50);
  if (latest != null) {
    String near = latest.dist > 0 ? round(latest.dist) + " cm" : "-";
    g.text("temp " + nf(latest.temp, 0, 1) + " C   humidity " + round(latest.hum) + " %   light " + int(latest.light)
      + "   sound " + int(latest.sound) + "   near " + near, 40, 72);
  } else if (!DEMO) {
    g.text("waiting for data from the Arduino...", 40, 72);
  }
  if (finished) {
    g.fill(RED);
    g.text("Complete. Saved " + savedName + ".png / .pdf    R = restart", 40, 94);
  }
  g.fill(INK, 110);
  g.text("R restart   S save   D demo", 40, H - 40);
}

// ---------------- Saving ----------------
void saveRecord() {
  String name = "exports/nest_" + year() + nf(month(), 2) + nf(day(), 2) + "_" + nf(hour(), 2) + nf(minute(), 2) + nf(second(), 2);
  String path = sketchPath(name);
  createPath(path);

  // High-resolution PNG
  PGraphics big = createGraphics(round(W * EXPORT_SCALE), round(H * EXPORT_SCALE));
  big.beginDraw();
  big.scale(EXPORT_SCALE);
  renderAll(big);
  big.endDraw();
  big.save(path + ".png");

  // Vector PDF (opens in Illustrator)
  PGraphics pdf = createGraphics(round(W), round(H), PDF, path + ".pdf");
  pdf.beginDraw();
  renderAll(pdf);
  pdf.dispose();
  pdf.endDraw();

  savedName = name;
  println("Saved " + name + ".png and " + name + ".pdf");
}

// Redraw the whole record from the stored readings
void renderAll(PGraphics g) {
  drawPaperAndGrid(g);
  Sample prev = null;
  for (Sample s : samples) {
    drawSample(g, prev, s);
    prev = s;
  }
}

// ---------------- Keys ----------------
void keyPressed() {
  if (key == 'r' || key == 'R') {
    restart();
  } else if (key == 's' || key == 'S') {
    saveRecord();
  } else if (key == 'd' || key == 'D') {
    DEMO = !DEMO;
    if (!DEMO && port == null) connect();
    restart();
  }
}
