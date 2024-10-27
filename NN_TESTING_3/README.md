# Project Title: Adaptive Object Recognition with Robot Kinematics

## Project Overview
This project involves enabling a humanoid robot, Hubert, to autonomously detect and manipulate plastic fruits within a designated workspace. The system utilizes a YOLOv8 model for object detection and a neural network for inverse kinematics, allowing Hubert to identify, locate, and pick up specific fruits. The project consists of multiple experiments to evaluate different optimizers and learning rates to achieve optimal performance for precise robot arm movement.

## Directory Structure

- **Hubert_Code/**: Contains the main code for controlling Hubert's movements and interactions.
  
- **experiment_adam_lr0.0002/**, **experiment_adam_lr0.001/**, **experiment_adam_lr0.01/**: Folders containing training and testing results for experiments using the Adam optimizer with different learning rates.

- **experiment_rmsprop_lr0.001/**: Folder with experiment results using the RMSprop optimizer with a learning rate of 0.001.

- **experiment_sgd_lr0.01/**, **experiment_sgd_lr0.1/**: Folders containing experiment results using the SGD optimizer with different learning rates.

- **ADAM_lr0.001.pth**: Pre-trained model file for the inverse kinematics neural network, trained with the Adam optimizer at a learning rate of 0.001.

- **Simulation_code.ipynb**: Jupyter Notebook containing the simulation code for visualizing Hubert's object detection and movement.

- **arduino_test_handin.py**: Python script that interfaces with the Arduino to control Hubert's servos for body, shoulder, and elbow movements based on detected fruit positions.

- **dataset.npz**: Dataset file containing training and testing data for the inverse kinematics neural network.

## Setup and Requirements

### Requirements
- Python 
- PyTorch
- OpenCV
- YOLOv8
- Arduino IDE (for uploading code to Hubert's servos)


