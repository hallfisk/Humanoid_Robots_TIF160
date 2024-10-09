import tkinter as tk
import serial
import time


# Initialize Serial Communication
ser = None  # Declare ser globally so it's accessible even if the connection fails
try:
    ser = serial.Serial('COM3', 9600, timeout=1)  # Replace 'COM3' with your correct port
    time.sleep(2)  # Give some time for the serial connection to establish
    print("Serial port opened successfully.")
except serial.SerialException:
    print("Could not open serial port. Check the connection.")

# Function to send servo command to Arduino
def send_command(servo_index, value):
    if ser.is_open:
        command = f"{servo_index} {value}\n"
        ser.write(command.encode())
        print(f"Sent: {command}")

# Function to update servo position based on slider value

def update_servo(index, scale):
    value = int(scale.get())
    send_command(index, value)
    entry_boxes[index].delete(0, tk.END)
    entry_boxes[index].insert(0, value)

# Function to update servo position based on manual entry
def manual_entry(index):
    try:
        value = int(entry_boxes[index].get())
        if pos_min[index] <= value <= pos_max[index]:
            sliders[index].set(value)
            send_command(index, value)
        else:
            print(f"Value out of range for {servo_names[index]} (must be between {pos_min[index]} and {pos_max[index]})")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

# Function to save current position to sequence
def save_sequence():
    current_positions = [slider.get() for slider in sliders]
    movement_sequence.append(current_positions)
    print(f"Saved sequence step: {current_positions}")

# Function to execute the saved sequence
def execute_sequence():
    for step in movement_sequence:
        for idx, position in enumerate(step):
            send_command(idx, position)
            sliders[idx].set(position)
            entry_boxes[idx].delete(0, tk.END)
            entry_boxes[idx].insert(0, position)
        time.sleep(1)  # Delay between each step

# Create Tkinter Window
root = tk.Tk()
root.title("Robot Joint Control")

# Define servo position limits
pos_min = [540, 1250, 550, 550]
pos_max = [2320, 2050, 2370, 2150]
servo_names = ['Body', 'Shoulder', 'Elbow', 'Gripper']

# List to hold the movement sequence
movement_sequence = []

# Create sliders and entry boxes for each servo
sliders = []
entry_boxes = []

for i, name in enumerate(servo_names):
    frame = tk.Frame(root)
    frame.pack()

    label = tk.Label(frame, text=name)
    label.pack()

    # Slider for servo
    slider = tk.Scale(frame, from_=pos_min[i], to=pos_max[i], orient=tk.HORIZONTAL, command=lambda value, idx=i: update_servo(idx, sliders[idx]))
    slider.pack()
    sliders.append(slider)

    # Entry box for manual input
    entry_box = tk.Entry(frame)
    entry_box.pack()
    entry_box.insert(0, (pos_min[i] + pos_max[i]) // 2)
    entry_boxes.append(entry_box)

    # Button to apply manual entry
    button = tk.Button(frame, text="Set Position", command=lambda idx=i: manual_entry(idx))
    button.pack()

# Set initial values on sliders
for idx, slider in enumerate(sliders):
    slider.set((pos_min[idx] + pos_max[idx]) // 2)  # Set to the middle of the min and max values

# Button to save current position to sequence
save_button = tk.Button(root, text="Save Current Position to Sequence", command=save_sequence)
save_button.pack()

# Button to execute the saved sequence
execute_button = tk.Button(root, text="Execute Saved Sequence", command=execute_sequence)
execute_button.pack()

# Run the GUI loop
root.mainloop()

# Close Serial Port when done
if ser.is_open:
    ser.close()
