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
const int pos_init[] = {1700, 1100, 1650, 1600}; // Initial positions for body, shoulder, elbow, and gripper
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

	delay(2000);



  // Set the desired positions for body, shoulder, elbow, and gripper
  int desired_body_pos = 1278;    // Example: set to 1002 microseconds
  int desired_shoulder_pos = 2298; // Example: set to 1233 microseconds
  int desired_elbow_pos = 605;   // Example: set to 1166 microseconds
  int desired_gripper_pos = 1000; // Example: desired gripper position

  // Move each servo incrementally to its desired position
  servo_body(desired_body_pos);
  delay(1000); // Small delay for movement
  
  servo_elbow(desired_elbow_pos);
  delay(200);

  servo_shoulder(desired_shoulder_pos);
  delay(1000);

  

  servo_gripper_ex(desired_gripper_pos);
  delay(1000);
}

void loop() {
  // No continuous behavior needed
}
