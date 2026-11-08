// uno-q-live — optional MCU companion sketch (Arduino UNO Q, STM32U585 side).
//
// The emotion classifier runs on the Linux MPU (see python/live_explain.py). This
// sketch does ONLY real-time I/O: it listens on the Bridge for a
// "set_emotion" RPC carrying the class index (0..8) and blinks/holds the
// onboard LED as a physical "how are they doing?" indicator.
//
//   happy (4)            -> LED solid on
//   surprise (8)         -> LED fast blink
//   angry (0), disgust (2), fear (3), sad (6) -> LED slow blink
//   everything else      -> LED off
//
// The app works with NO sketch loaded (Bridge calls from Python are
// best-effort and never fatal). This file has not been compiled or flashed
// in this environment — treat it as a starting point and verify in App Lab.

#include "Arduino_RouterBridge.h"

const int LED_PIN = LED_BUILTIN;

// Canonical FANE class order:
// 0 angry, 1 confused, 2 disgust, 3 fear, 4 happy, 5 neutral, 6 sad, 7 shy, 8 surprise
volatile int emotion = -1;  // -1 = no reading yet
unsigned long lastToggle = 0;
bool ledOn = false;

void set_emotion(int cls) {
  if (cls >= 0 && cls <= 8) {
    emotion = cls;
  }
}

// Returns blink half-period in ms: 0 = solid on, -1 = off.
int ledPatternMs() {
  switch (emotion) {
    case 4:  return 0;    // happy: solid
    case 8:  return 120;  // surprise: fast blink
    case 0:               // angry
    case 2:               // disgust
    case 3:               // fear
    case 6:  return 600;  // sad: slow blink
    default: return -1;   // confused/neutral/shy/none: off
  }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);
  Bridge.begin();
  Bridge.provide("set_emotion", set_emotion);
}

void loop() {
  int period = ledPatternMs();
  if (period == 0) {
    digitalWrite(LED_PIN, HIGH);
  } else if (period < 0) {
    digitalWrite(LED_PIN, LOW);
  } else {
    unsigned long now = millis();
    if (now - lastToggle >= (unsigned long)period) {
      lastToggle = now;
      ledOn = !ledOn;
      digitalWrite(LED_PIN, ledOn ? HIGH : LOW);
    }
  }
  delay(10);
}
