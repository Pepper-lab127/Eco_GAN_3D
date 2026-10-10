# Goose Nest Guardian

An Arduino nest that judges whether its surroundings suit a Canada goose egg
and shows how it feels with a fan of five grass "tail feathers" on servos. When
something approaches, the same feathers rise up, rattle and swat at it.

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

Aim the left distance and motion sensors about 30° to the left and the right
pair about 30° to the right. Their beams should overlap a little in front of
the nest. The nest uses that layout to tell which side something is coming from.

Power the servos from their own 5–6 V supply rated for at least 2–3 A, and join
its ground to the Arduino's ground.

Libraries: **DHT sensor library** (Adafruit), **Adafruit Unified Sensor**, Servo.

## What the feathers mean

| Feather | Shows |
|---|---|
| 0 (far left) | temperature |
| 1 (left) | humidity |
| 2 (center) | overall comfort |
| 3 (right) | light |
| 4 (far right) | quietness |

A tall feather means its factor is good and a drooping one means it is poor.

- **Content**: a slow ripple runs across the fan.
- **Uneasy**: the feather for the worst factor twitches now and then.
- **Distressed**: the fan sags and shivers.

| Threat | Trigger | Reaction |
|---|---|---|
| ALERT | motion, something within 2 m, or a sudden loud noise | fan rises, leans toward it, quivers |
| WARN | within 1 m, approaching fast, or motion lasting more than 4 s | full display, rattle, hiss |
| DEFEND | within 40 cm | feathers on that side swat fast |

Each threat level drops back one step after 4 s of calm.

## Demo mode

`DEMO_MODE = true` (the default) is for showing the nest indoors. A normal room
(19–25 °C, 35–65 % humidity, the room's own light level) counts as comfortable,
so the fan starts out content. People can then change its mood:

| Do this | What happens |
|---|---|
| Cup a hand around the DHT11 | At about 28 °C the nest turns uneasy, and at 30 °C distressed |
| Breathe on the DHT11 | Humidity rises and the humidity feather droops |
| Shine a phone torch on the photoresistor | The light feather droops |
| Talk loudly or clap near the KY-037 | The quiet feather droops, and a sudden clap startles it into ALERT |

The nest measures the room's light during the first 5 seconds and treats that
as normal, so don't shade or light the sensor while it starts up.

Set `DEMO_MODE = false` to use the real goose-egg targets (37–38 °C).

## First-time setup

1. Set `FEATHER_DOWN_ANGLE` / `FEATHER_UP_ANGLE` for each servo before you
   attach the grass. If a servo is mounted mirrored, swap its two values.
2. **HC-SR501**: put the jumper on **H** (repeat trigger), turn the time knob
   fully anticlockwise (shortest hold, about 3 s) and set the sensitivity knob
   to the middle.
3. **KY-037**: open the Serial Monitor (9600 baud) and watch `noise=`. Read it
   once in a quiet room and once while someone talks nearby, then set
   `NOISE_IDEAL_HI` between the two. If the nest gets startled by ordinary
   noises, raise `LOUD_MARGIN`. Keep the microphone away from the servos.
4. Set `LOCKDOWN_PHASE = true` for the last 3 days before hatching. This raises
   the humidity target.
5. The motion sensors are ignored for the first 60 s while they warm up.

## DHT11 limits

The DHT11 reads temperature in whole degrees, accurate to ±2 °C, and humidity
from 20–80 %, accurate to ±5 %. That's enough for this project, but too rough
to control a real incubator. For that, a DHT22 drops straight in: change
`DHT_TYPE` to `DHT22`.
