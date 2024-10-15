#include <Arduino.h>
#include <Servo.h>

// Create servo objects
Servo servo1;
Servo servo2;

// Variables to store the angles for the servos
int angle1 = 90;  // Default angle for servo 1
int angle2 = 90;  // Default angle for servo 2

void setup() {
  Serial.begin(9600);  // Start serial communication at 9600 baud rate
  
  // Attach the servos to the PWM pins
  servo1.attach(11);   // Servo 1 on pin 9
  servo2.attach(10);  // Servo 2 on pin 10
  
  // Move the servos to the default position
  servo1.write(angle1);
  servo2.write(angle2);
}

void loop() {
  if (Serial.available() > 0) {
    // Read the angle values for both servos from the serial input
    String input = Serial.readStringUntil('\n');  // Read input until newline
    input.trim();  // Remove any trailing spaces or newlines

    if (input.startsWith("S1:")) {
      // Input for servo 1 in the format "S1:angle"
      angle1 = input.substring(3).toInt();  // Extract the angle
      angle1 = constrain(angle1, 0, 180);  // Constrain the angle between 0 and 180 degrees
      servo1.write(angle1);  // Move servo 1
    }
    else if (input.startsWith("S2:")) {
      // Input for servo 2 in the format "S2:angle"
      angle2 = input.substring(3).toInt();  // Extract the angle
      angle2 = constrain(angle2, 0, 180);  // Constrain the angle between 0 and 180 degrees
      servo2.write(angle2);  // Move servo 2
    }
  }
  // servo1.write(180);
  // servo2.write(200);
  // delay(1000);
}
