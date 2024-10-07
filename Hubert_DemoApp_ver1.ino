#include <Arduino.h>
#include <Servo.h>

// Servos
Servo body;
Servo shoulder;
Servo elbow;
Servo gripper;

// Servo pins
const int servo_pins[] = {3, 9, 10, 11}; // Pins for body, shoulder, elbow, and gripper

const int pos_min[] = {560, 550, 950, 750, 550, 550};
const int pos_max[] = {2330, 2500, 2400, 2150};

// Init position of all servos
const int pos_init[] = {1430, 1100, 1650, 1000}; // Initial positions for body, shoulder, elbow, and gripper
int curr_pos[4]; // Stores current positions for body, shoulder, elbow, and gripper
int new_servo_val[4];

// Function to move the body servo incrementally
void servo_body(int new_pos) {
  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  // Current servo value
  now = curr_pos[0];
  CurrPwm = now;
  NewPwm = new_pos;

  // Determine direction of movement
  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
      now = now + delta*diff;
      body.writeMicroseconds(now);
      delay(20);
    }
    curr_pos[0] = now;
    delay(10);

}


void servo_shoulder(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[1];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    shoulder.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[1] = now;
  delay(10);
}

//Servo update function
void servo_elbow(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[2];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    elbow.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[2] = now;
  delay(10);
}

void servo_gripper_ex(const int new_pos) {

  int diff, steps, now, CurrPwm, NewPwm, delta = 6;

  //current servo value
  now = curr_pos[3];
  CurrPwm = now;
  NewPwm = new_pos;

  /* determine interation "diff" from old to new position */
  diff = (NewPwm - CurrPwm)/abs(NewPwm - CurrPwm); // Should return +1 if NewPwm is bigger than CurrPwm, -1 otherwise.
  steps = abs(NewPwm - CurrPwm);
  delay(10);

  for (int i = 0; i < steps; i += delta) {
    now = now + delta*diff;
    gripper.writeMicroseconds(now);
    delay(20);
  }
  curr_pos[3] = now;
  delay(10);
}



void setup() {
  // Attach each servo to its corresponding pin
  body.attach(servo_pins[0]);
  body.writeMicroseconds(pos_init[0]);
  shoulder.attach(servo_pins[1]);
  shoulder.writeMicroseconds(pos_init[1]);
  elbow.attach(servo_pins[2]);
  elbow.writeMicroseconds(pos_init[2]);
  gripper.attach(servo_pins[3]);
  gripper.writeMicroseconds(pos_init[3]);

  //Initilize curr_pos and new_servo_val vectors
  byte i;
  for (i=0; i<(sizeof(pos_init)/sizeof(int)); i++){
    curr_pos[i] = pos_init[i];
    new_servo_val[i] = curr_pos[i];
  }

  Serial.begin(57600);
  
  /*
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim(); // Remove any whitespace or newline characters
    Serial.println("Received data: " + data);

    // Split the data by commas
    int index1 = data.indexOf(',');
    int index2 = data.indexOf(',', index1 + 1);

    if (index1 == -1 || index2 == -1) {
      Serial.println("Error: Received data in wrong format.");
      return;
    }

    String body_str = data.substring(0, index1);
    String shoulder_str = data.substring(index1 + 1, index2);
    String elbow_str = data.substring(index2 + 1);

    float desired_body_pos = body_str.toFloat();
    float desired_shoulder_pos = shoulder_str.toFloat();
    float desired_elbow_pos = elbow_str.toFloat();

    Serial.print("Body: ");
    Serial.println(desired_body_pos);
    Serial.print("Shoulder: ");
    Serial.println(desired_shoulder_pos);
    Serial.print("Elbow: ");
    Serial.println(desired_elbow_pos);

    int desired_gripper_pos = 1400; // Example: desired gripper positiom

    // Move arm above box
    servo_elbow(600);
    delay(500);
    servo_shoulder(2100);
    delay(500);

    // Move to fruit pos
    servo_body(desired_body_pos);
    delay(500);
    servo_elbow(desired_elbow_pos);
    delay(500);
    servo_shoulder(desired_shoulder_pos);
    delay(500);
    servo_gripper_ex(desired_gripper_pos);
    delay(500);
  }
  */
  
}



void loop() {
  //Serial.print("Arduino is ready.");
  int a = 0;
  if (Serial.available() > 0 && a == 0) {
    a = 1;
    String data = Serial.readStringUntil('\n');
    data.trim(); // Remove any whitespace or newline characters
    Serial.println("Received data: " + data);

    // Split the data by commas
    int index1 = data.indexOf(',');
    int index2 = data.indexOf(',', index1 + 1);

    if (index1 == -1 || index2 == -1) {
      Serial.println("Error: Received data in wrong format.");
      return;
    }

    String body_str = data.substring(0, index1);
    String shoulder_str = data.substring(index1 + 1, index2);
    String elbow_str = data.substring(index2 + 1);

    float desired_body_pos = body_str.toFloat();
    float desired_shoulder_pos = shoulder_str.toFloat();
    float desired_elbow_pos = elbow_str.toFloat();

    Serial.print("Body: ");
    Serial.println(desired_body_pos);
    Serial.print("Shoulder: ");
    Serial.println(desired_shoulder_pos);
    Serial.print("Elbow: ");
    Serial.println(desired_elbow_pos);

    int desired_gripper_pos = 1400; // Example: desired gripper positiom

    // Move arm above box
    servo_elbow(600);
    delay(500);
    servo_shoulder(2100);
    delay(500);

    // Move to fruit pos
    servo_body(desired_body_pos);
    delay(500);
    servo_elbow(desired_elbow_pos);
    delay(500);
    servo_shoulder(desired_shoulder_pos);
    delay(500);
    servo_gripper_ex(desired_gripper_pos);
    delay(500);
  }
}
