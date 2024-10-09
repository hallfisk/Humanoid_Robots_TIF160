#include <Servo.h>

// Define the servos
Servo body;
Servo shoulder;
Servo elbow;
Servo gripper;

// Servo pins
const int servo_pins[] = {3, 9, 10, 11}; // Pins for body, shoulder, elbow, and gripper
int curr_pos[] = {1430, 1100, 1650, 1000};  // Initial positions

// Initialize variables for serial input
String inputString = "";
bool stringComplete = false;

void setup() {
  // Initialize the serial communication
  Serial.begin(9600);

  // Attach servos
  body.attach(servo_pins[0]);
  shoulder.attach(servo_pins[1]);
  elbow.attach(servo_pins[2]);
  gripper.attach(servo_pins[3]);

  // Set initial positions
  body.writeMicroseconds(curr_pos[0]);
  shoulder.writeMicroseconds(curr_pos[1]);
  elbow.writeMicroseconds(curr_pos[2]);
  gripper.writeMicroseconds(curr_pos[3]);

  // Clear the input string
  inputString.reserve(200);
}

// Function to move the body servo incrementally
void servo_body(int new_pos) {
  int diff, steps, now, CurrPwm, NewPwm, delta = 6;
  now = curr_pos[0];
  CurrPwm = now;
  NewPwm = new_pos;

  // Determine the direction of movement
  diff = (NewPwm - CurrPwm) / abs(NewPwm - CurrPwm); // +1 if NewPwm > CurrPwm, -1 otherwise
  steps = abs(NewPwm - CurrPwm);

  delay(10);
  for (int i = 0; i < steps; i += delta) {
    now = now + delta * diff;
    body.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[0] = now;
  delay(10);
}

// Function to move the shoulder servo incrementally
void servo_shoulder(int new_pos) {
  int diff, steps, now, CurrPwm, NewPwm, delta = 6;
  now = curr_pos[1];
  CurrPwm = now;
  NewPwm = new_pos;

  // Determine the direction of movement
  diff = (NewPwm - CurrPwm) / abs(NewPwm - CurrPwm); // +1 if NewPwm > CurrPwm, -1 otherwise
  steps = abs(NewPwm - CurrPwm);

  delay(10);
  for (int i = 0; i < steps; i += delta) {
    now = now + delta * diff;
    shoulder.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[1] = now;
  delay(10);
}

// Function to move the elbow servo incrementally
void servo_elbow(int new_pos) {
  int diff, steps, now, CurrPwm, NewPwm, delta = 6;
  now = curr_pos[2];
  CurrPwm = now;
  NewPwm = new_pos;

  // Determine the direction of movement
  diff = (NewPwm - CurrPwm) / abs(NewPwm - CurrPwm); // +1 if NewPwm > CurrPwm, -1 otherwise
  steps = abs(NewPwm - CurrPwm);

  delay(10);
  for (int i = 0; i < steps; i += delta) {
    now = now + delta * diff;
    elbow.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[2] = now;
  delay(10);
}

// Function to move the gripper incrementally
void servo_gripper_ex(int new_pos) {
  int diff, steps, now, CurrPwm, NewPwm, delta = 6;
  now = curr_pos[3];
  CurrPwm = now;
  NewPwm = new_pos;

  // Determine the direction of movement
  diff = (NewPwm - CurrPwm) / abs(NewPwm - CurrPwm); // +1 if NewPwm > CurrPwm, -1 otherwise
  steps = abs(NewPwm - CurrPwm);

  delay(10);
  for (int i = 0; i < steps; i += delta) {
    now = now + delta * diff;
    gripper.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[3] = now;
  delay(10);
}

// Function to parse and process serial commands
void processInput() {
  if (stringComplete) {
    // Split the input string into servo index and value
    int spaceIndex = inputString.indexOf(' ');
    if (spaceIndex > 0) {
      int servoIndex = inputString.substring(0, spaceIndex).toInt();
      int servoValue = inputString.substring(spaceIndex + 1).toInt();

      // Move the servo incrementally based on the index
      switch (servoIndex) {
        case 0:
          servo_body(servoValue);
          break;
        case 1:
          servo_shoulder(servoValue);
          break;
        case 2:
          servo_elbow(servoValue);
          break;
        case 3:
          servo_gripper_ex(servoValue);
          break;
        default:
          Serial.println("Invalid servo index.");
      }

      Serial.print("Moving servo ");
      Serial.print(servoIndex);
      Serial.print(" to ");
      Serial.println(servoValue);
    }

    // Clear the input string and reset
    inputString = "";
    stringComplete = false;
  }
}

void loop() {
  // Listen for serial input
  if (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      stringComplete = true;
    } else {
      inputString += inChar;
    }
  }

  // Process the input string when it's complete
  processInput();
}
