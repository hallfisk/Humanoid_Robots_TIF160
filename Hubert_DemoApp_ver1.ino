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
const int pos_init[] = {2300, 2050, 550, 600}; // Initial positions for body, shoulder, elbow, and gripper
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

void move_shoulder_and_elbow(const int new_pos_shoulder, const int new_pos_elbow) {

  int diff_shoulder, diff_elbow, steps_shoulder, steps_elbow;
  int now_shoulder, now_elbow, CurrPwm_shoulder, CurrPwm_elbow, NewPwm_shoulder, NewPwm_elbow;
  int delta = 6;

  // current servo values
  now_shoulder = curr_pos[1];
  now_elbow = curr_pos[2];
  
  CurrPwm_shoulder = now_shoulder;
  CurrPwm_elbow = now_elbow;

  NewPwm_shoulder = new_pos_shoulder;
  NewPwm_elbow = new_pos_elbow;

  // Determine direction (diff) and steps for shoulder
  diff_shoulder = (NewPwm_shoulder - CurrPwm_shoulder) / abs(NewPwm_shoulder - CurrPwm_shoulder);
  steps_shoulder = abs(NewPwm_shoulder - CurrPwm_shoulder);

  // Determine direction (diff) and steps for elbow
  diff_elbow = (NewPwm_elbow - CurrPwm_elbow) / abs(NewPwm_elbow - CurrPwm_elbow);
  steps_elbow = abs(NewPwm_elbow - CurrPwm_elbow);

  delay(10);

  // Move both servos together, synchronizing their steps
  for (int i = 0; i < max(steps_shoulder, steps_elbow); i += delta) {
    
    // Update shoulder position
    if (i < steps_shoulder) {
      now_shoulder += delta * diff_shoulder;
      shoulder.writeMicroseconds(now_shoulder);
    }

    // Update elbow position
    if (i < steps_elbow) {
      now_elbow += delta * diff_elbow;
      elbow.writeMicroseconds(now_elbow);
    }

    delay(20);
  }

  // Update current positions
  curr_pos[1] = now_shoulder;
  curr_pos[2] = now_elbow;

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
  shoulder.attach(servo_pins[1]);
  elbow.attach(servo_pins[2]);
  gripper.attach(servo_pins[3]);

  body.writeMicroseconds(pos_init[0]);
  shoulder.writeMicroseconds(pos_init[1]);
  elbow.writeMicroseconds(pos_init[2]);
  gripper.writeMicroseconds(pos_init[3]);
  
  //Initilize curr_pos and new_servo_val vectors
  byte i;
  for (i=0; i<(sizeof(pos_init)/sizeof(int)); i++){
    curr_pos[i] = pos_init[i];
    new_servo_val[i] = curr_pos[i];
  }

  Serial.begin(57600);
  
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    data.trim();
    Serial.println("Received data: " + data);

    // Split the data by commas
    int index1 = data.indexOf(',');
    int index2 = data.indexOf(',', index1 + 1);
    int index3 = data.indexOf(',', index2 + 1);

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

    // Move the body, shoulder, and elbow to the positions
    servo_body(desired_body_pos);
    delay(500);
    move_shoulder_and_elbow(desired_shoulder_pos, desired_elbow_pos);
    //delay(500);

    while (true) {
      if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n');
        command.trim();

        if (command == "close") {
          // Close the gripper
          servo_gripper_ex(1400);  // Adjust to the value for "close"
          delay(500);

          // Move shoulder, elbow, and body to new positions
          move_shoulder_and_elbow(2050, 550);  // Move shoulder and elbow
          delay(500);
          servo_body(1800);  // Move body
          delay(500);

          // Open the gripper
          servo_gripper_ex(600);  // Adjust to the value for "open"
          Serial.println("Task completed.");
          break;  // Exit the loop and finish the task
        }
      }
    }
  }
}


/*
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

    // Move to fruit pos
    servo_body(desired_body_pos);
    delay(500);
    move_shoulder_and_elbow(desired_shoulder_pos, desired_elbow_pos);
    delay(500);
    //servo_gripper_ex(desired_gripper_pos);
    //delay(500);

    Serial.print("DONE");
  }

  if (Serial.available() > 0 && a == 0) {

  }
}
*/