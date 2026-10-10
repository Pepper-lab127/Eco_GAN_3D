# Goose Nest Guardian

An Arduino nest that senses its surroundings and shows them with a fan of five
grass "tail feathers" on servos. Each feather follows one sensor.

## Wiring

| Part | Pin |
|---|---|
| 5 servos, signal (far left → far right) | D8, D9, D10, D11, D12 |
| HC-SR04 distance sensor, left (TRIG / ECHO) | D7 / D6 |
| HC-SR04 distance sensor, right (TRIG / ECHO) | D5 / D4 |
| HC-SR501 motion sensor: left / right (OUT) | D3 / D2 |
| Photoresistor + 10k divider | A0 |
| KY-037 sound sensor (AO) | A1 |
| Passive buzzer (optional) | A2 |
| DHT11 temperature/humidity (S) | A3 |

Open `wiring.html` in a browser for the full wiring diagram.

Every sensor's VCC goes to the Arduino's 5V and its GND to GND. The KY-037's DO
pin is not used.

Power the servos from their own 5–6 V supply rated for at least 2–3 A, and join
its ground to the Arduino's ground.

Libraries: **DHT sensor library** (Adafruit), **Adafruit Unified Sensor**, Servo.

## What the feathers do

Each feather follows one sensor reading, moving between fully down and fully up:

| Feather | Follows | Down | Up |
|---|---|---|---|
| 0 (far left) | temperature | 18 °C | 32 °C |
| 1 (left) | humidity | 30 % | 85 % |
| 2 (center) | distance (closer of the two sensors) | 150 cm or more | 10 cm |
| 3 (right) | light | dark | bright |
| 4 (far right) | sound | quiet | loud |

The motion sensors and the buzzer can stay wired up, but this version doesn't use them.

## Setting it up

1. Set `DOWN_ANGLE` / `UP_ANGLE` for each servo before you attach the grass.
   If a servo is mounted the other way round, swap its two numbers.
2. Open the Serial Monitor (9600 baud). It prints every reading and every
   servo angle twice a second.
3. Change the ranges at the top of the sketch (`TEMP_LOW`, `TEMP_HIGH` and so
   on) so the feathers move as much as you want in your room. For example, if
   `sound` sits around 20 when it's quiet and reaches 80 when you clap, set
   `SOUND_LOW = 20` and `SOUND_HIGH = 80`.
4. If the feathers jitter, lower `SMOOTHING`. If they react too slowly, raise it.

## DHT11 limits

The DHT11 reads temperature in whole degrees, accurate to ±2 °C, and humidity
from 20–80 %, accurate to ±5 %. That's enough for this project, but too rough
to control a real incubator. For that, a DHT22 drops straight in: change
`DHT11` to `DHT22` in the line `DHT dht(PIN_DHT, DHT11);`.
