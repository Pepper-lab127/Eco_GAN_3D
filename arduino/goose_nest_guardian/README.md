# Goose Nest Guardian

An Arduino nest that judges whether its surroundings suit a Canada goose egg
and shows how it feels with a fan of five grass "tail feathers" on servos. When
something approaches, the same feathers rise up, rattle and swat at it.

## Parts

| Part | Pin |
|---|---|
| DHT22 temperature/humidity | D2 |
| PIR motion sensor: left / center / right | D4 / D5 / D6 |
| HC-SR04 ultrasonic (TRIG / ECHO), facing straight out | D7 / D8 |
| 5 servos (far left → far right) | D3, D9, D10, D11, D12 |
| Photoresistor + 10k divider | A0 |
| MQ-135 air quality (AO) | A1 |
| Passive buzzer (optional) | A2 |

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
| 4 (far right) | air quality |

A tall feather means its factor is good and a drooping one means it is poor.

- **Content**: a slow ripple runs across the fan.
- **Uneasy**: the feather for the worst factor twitches now and then.
- **Distressed**: the fan sags and shivers.

| Threat | Trigger | Reaction |
|---|---|---|
| ALERT | motion, or something within 2 m | fan rises, leans toward it, quivers |
| WARN | within 1 m, approaching fast, or motion lasting more than 4 s | full display, rattle, hiss |
| DEFEND | within 40 cm | feathers on that side swat fast |

Each threat level drops back one step after 4 s of calm.

## First-time setup

1. Set `FEATHER_DOWN_ANGLE` / `FEATHER_UP_ANGLE` for each servo before you
   attach the grass. If a servo is mounted mirrored, swap its two values.
2. Open the Serial Monitor (9600 baud) and look at the `air=` reading in clean
   air. Set `AIR_IDEAL_HI` a little above that value.
3. Set `LOCKDOWN_PHASE = true` for the last 3 days before hatching. This raises
   the humidity target.
4. The PIR sensors are ignored for the first 30 s and the air sensor for the
   first 60 s while they warm up.
