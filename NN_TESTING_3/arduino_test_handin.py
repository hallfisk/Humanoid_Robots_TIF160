import serial
import time
import cv2
from ultralytics import YOLO
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Load YOLOv8 large model
model = YOLO('yolov8l.pt')
model.to('cpu')

def detect_fruit_and_save_image(image_path, target_fruit='apple', output_path='output_image.jpg'):
    # Check if the file exists
    if not os.path.isfile(image_path):
        print(f"Error: The file '{image_path}' does not exist.")
        return None

    # Load the image
    img = cv2.imread(image_path)

    # Check if the image was loaded correctly
    if img is None:
        print(f"Error: Failed to load the image from '{image_path}'.")
        return None

    # Resize the image for YOLO detection (optional)
    img_resized = cv2.resize(img, (1000, 600))

    #img_resized = cv2.rotate(img_resized, cv2.ROTATE_180)

    def get_coordinates(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
            print(f"Coordinates: ({x}, {y})")  # Print the coordinates
            coordinates_of_box.append((x, y))  # Save the coordinates in a list

    coordinates_of_box = []

    # MAKE SURE TO CLICK IN THIS ORDER: TOP-LEFT -> TOP-RIGHT -> BOTTOM-RIGHT -> BOTTOM-LEFT -> ANY KEY TO ESCAPE
    cv2.imshow('Image', img_resized)
    cv2.setMouseCallback('Image', get_coordinates)  # Set the callback function
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Use the YOLO model to detect objects
    results = model(img_resized)

    # Get class names
    class_names = model.names

    # Find the class ID for the target fruit
    target_fruit_class_id = None
    for class_id, name in class_names.items():
        if name == target_fruit:
            target_fruit_class_id = class_id
            break

    if target_fruit_class_id is None:
        print(f"Error: The target fruit '{target_fruit}' is not in the YOLO dataset.")
        return None

    target_fruit_positions = []

    # Loop over detections
    for detection in results[0].boxes:
        class_id = int(detection.cls)  # Class ID of the detected object

        # Get bounding box coordinates
        xmin, ymin, xmax, ymax = map(int, detection.xyxy[0])
        # Calculate the center of the bounding box
        x_center = (xmin + xmax) // 2
        y_center = (ymin + ymax) // 2

        # Draw a red dot if the detected fruit is the target fruit, else blue dot
        if class_id == target_fruit_class_id:
            cv2.circle(img_resized, (x_center, y_center), 10, (0, 0, 255), -1)  # Red dot for target fruit
            target_fruit_positions.append((x_center, y_center))
        elif class_id in [46, 47, 49, 50, 51]:  # IDs corresponding to fruits (banana, apple, orange, broccoli, carrot)
            cv2.circle(img_resized, (x_center, y_center), 10, (255, 0, 0), -1)  # Blue dot for other fruits

    for (x, y) in coordinates_of_box:
        cv2.circle(img_resized, (x, y), 10, (0, 255, 0), -1)

    
    # Save the detection image
    cv2.imwrite(output_path, img_resized)

    if target_fruit_positions:
        print(f"Detected {target_fruit} at positions: {target_fruit_positions}")
    else:
        print(f"No {target_fruit} detected in the image.")

    x_zero, y_zero = coordinates_of_box[2] #bottom-right corner of box
    x_fruit, y_fruit = target_fruit_positions[0]

    box_size_x_mm = 282 #mm (measured IRL)
    box_size_y_mm = 240 #mm (measured IRL)
    box_size_diag_mm = (box_size_x_mm**2 + box_size_y_mm**2)**0.5 #mm (measured IRL)

    # Calculate distances from picture in x
    box_size_x_pixels_1 = np.linalg.norm(np.array(coordinates_of_box[0])-np.array(coordinates_of_box[1])) #mm (width in pixels from picture)
    box_size_x_pixels_2 = np.linalg.norm(np.array(coordinates_of_box[2])-np.array(coordinates_of_box[3])) #mm (width in pixels from picture)

    # Calculate distances from picture in y
    box_size_y_pixels_1 = np.linalg.norm(np.array(coordinates_of_box[0])-np.array(coordinates_of_box[3])) #mm (width in pixels from picture)
    box_size_y_pixels_2 = np.linalg.norm(np.array(coordinates_of_box[1])-np.array(coordinates_of_box[2])) #mm (width in pixels from picture)

    # Calculate distances from picture in diagonals
    box_size_diag_pixels_1 = np.linalg.norm(np.array(coordinates_of_box[0])-np.array(coordinates_of_box[2])) #mm (width in pixels from picture)
    box_size_diag_pixels_2 = np.linalg.norm(np.array(coordinates_of_box[1])-np.array(coordinates_of_box[3])) #mm (width in pixels from picture)

    # Get mean distances for x and y
    box_size_x_pixels_mean = (box_size_x_pixels_1 + box_size_x_pixels_2) / 2
    box_size_y_pixels_mean = (box_size_y_pixels_1 + box_size_y_pixels_2) / 2
    box_size_diag_pixels_mean = (box_size_diag_pixels_1 + box_size_diag_pixels_2) / 2

    # Get mm per pixel for x and y
    mm_per_pixel_x = box_size_x_mm / box_size_x_pixels_mean # mm / pixel
    mm_per_pixel_y = box_size_y_mm / box_size_y_pixels_mean # mm / pixel
    mm_per_pixel_diag = box_size_diag_mm / box_size_diag_pixels_mean # mm / pixel

    # Get overall mean mm per pixel
    mm_per_pixel = (mm_per_pixel_x + mm_per_pixel_y + mm_per_pixel_diag) / 3
    
    relative_x_fruit_pixels = abs(x_fruit - x_zero)
    relative_y_fruit_pixels = abs(y_fruit - y_zero)

    relative_x_fruit_mm = relative_x_fruit_pixels*mm_per_pixel
    relative_y_fruit_mm = relative_y_fruit_pixels*mm_per_pixel
    
    print(f'\nPixels from {target_fruit} to bottom-right corner:')
    print('x (pixels):', relative_x_fruit_pixels)
    print('x (mm):', relative_x_fruit_mm)
    print('y (pixles):', relative_y_fruit_pixels)
    print('y (mm):', relative_y_fruit_mm)

    return relative_x_fruit_mm, relative_y_fruit_mm

# 
image_path = 'B_and_C.jpg'  # Replace with image path
output_image = 'box_with_fruits_and_edges_marked.jpg'  # Path to save the image with marked dots

fruits_list = ['banana', 'apple', 'orange']
fruit_index = int(input('Choose index following [0, 1, 2] = [banana, apple, orange]: '))
target_fruit = fruits_list[fruit_index]  # Specify the fruit to mark with red dots0

relative_x_fruit_mm, relative_y_fruit_mm = detect_fruit_and_save_image(image_path, target_fruit, output_image)

# Distances from bottom-right corner of box to the base system of Hubert
box_bottom_right_corner_to_base_system_x = 60 #mm1
box_bottom_right_corner_to_base_system_y = -140  #mm
box_bottom_right_corner_to_base_system_z = 94 #mm

if fruit_index == 0: #banana
    relative_z_fruit_mm = 70 #start with set value to begin with
else: #apple or orange
    relative_z_fruit_mm = 70

fruit_position_base_system_x = box_bottom_right_corner_to_base_system_x + relative_y_fruit_mm
fruit_position_base_system_y = box_bottom_right_corner_to_base_system_y + relative_x_fruit_mm
fruit_position_base_system_z = box_bottom_right_corner_to_base_system_z + relative_z_fruit_mm

fruit_position_base_system = [fruit_position_base_system_x, fruit_position_base_system_y, fruit_position_base_system_z]

print("\nFruit pos base system (mm): ", fruit_position_base_system)

class InverseKinematicsNN(nn.Module):
    def __init__(self):
        super(InverseKinematicsNN, self).__init__()
        self.fc1 = nn.Linear(3, 64)  # Input layer (x, y, z) -> 64 neurons
        self.fc2 = nn.Linear(64, 128)  # Hidden layer -> 128 neurons
        self.fc3 = nn.Linear(128, 256)  # Hidden layer -> 128 neurons
        self.fc4 = nn.Linear(256, 128)  # Hidden layer -> 128 neurons
        self.fc5 = nn.Linear(128, 64)  # Hidden layer -> 64 neurons
        self.fc6 = nn.Linear(64, 3)   # Output layer -> (theta_1, theta_2, theta_3)
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = self.leaky_relu(self.fc2(x))
        x = self.leaky_relu(self.fc3(x))
        x = self.leaky_relu(self.fc4(x))
        x = self.leaky_relu(self.fc5(x))
        x = self.fc6(x)  # Output angles in radians
        return x

model = InverseKinematicsNN()
model.load_state_dict(torch.load('best_model_latest.pth'))

test_position = torch.tensor(fruit_position_base_system, dtype=torch.float32)/1000
print('\nTest pos (m): ', test_position)
print()

# Put your model in evaluation mode
model.eval()

# Make prediction
with torch.no_grad():
    predicted_angles = model(test_position)  # Predict the joint angles from the model
    predicted_angles_rad = predicted_angles.detach().numpy()

print('Predicted angles [rad]: ', predicted_angles_rad)
predicted_angles_deg = predicted_angles_rad*180/np.pi
print('Predicted angles [deg]: ', predicted_angles_deg)
print()

def angle_to_millisec(ls): #body, shoulder, elbow
    theta1, theta2, theta3 = ls
    
    body_servo = 540 + ((-theta1+90)/180) * (2320 - 540) 
    shoulder_servo = 1250 + (theta2/90 * (2050 - 1250))
    elbow_servo = 550 + ((theta3+90)/180 * (2370 - 550))
    
     # Clamp the servo positions to their respective limits
    body_servo = max(540, min(body_servo, 2320))  # Min: 540, Max: 2320
    shoulder_servo = max(1250, min(shoulder_servo, 2050))  # Min: 1250, Max: 2050
    elbow_servo = max(550, min(elbow_servo, 2370))  # Min: 550, Max: 2370

    return body_servo, shoulder_servo, elbow_servo

body_servo, shoulder_servo, elbow_servo = angle_to_millisec(predicted_angles_deg)

print('Body servo [ms]: ', body_servo)
print('Shoulder servo [ms]: ', shoulder_servo)
print('Elbow servo [ms]: ', elbow_servo)
print()

import serial
import time

def send_command(arduino, body, shoulder, elbow, gripper=None):
    # Send the servo values to the Arduino as a single string
    servo_data = f"{body},{shoulder},{elbow}"
    if gripper is not None:
        servo_data += f",{gripper}"  # Add gripper value if specified
    servo_data += "\n"
    arduino.write(servo_data.encode('utf-8'))
    time.sleep(1)  # Give some time for Arduino to process the command

# Establish serial connection
port_name = '/dev/cu.usbmodem101'
arduino = serial.Serial('COM3', baudrate=57600, timeout=.2)
arduino.dtr = False  # Prevent Arduino reset on connection
time.sleep(2)  # Delay to stabilize serial connection

# Send the initial servo positions to the Arduino
send_command(arduino, body_servo, shoulder_servo, elbow_servo)

# Wait for Arduino response (optional, can be removed if not needed)
response = arduino.readline().decode('utf-8').strip()
if response:
    print("Arduino says:", response)

# Wait for manual input from the user
while True:
    command = input("Enter 'close' to close the gripper and move the robot: ")
    if command.lower() == "close":

        print('OK')
        
        command += "\n"
        arduino.write(command.encode('utf-8'))
        time.sleep(5)
        
        break  # End the loop and finish the program
    else:
        print("Unknown command. Please type 'close' to continue.")