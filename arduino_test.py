import serial
import time

port_name = '/dev/cu.usbmodem1101'
arduino = serial.Serial(port=port_name, baudrate=57600, timeout=.1)

while True:
    time.sleep(0.1)  # Slight delay to avoid overwhelming the Arduino
    data = arduino.readline().decode('utf-8').strip()
    if "Input either 1 or 0" in data:
        x = input("Enter 1 to execute movement or 0 to wait: ")  # Taking input from user
        arduino.write(bytes(x, 'utf-8'))  # Send input to Arduino
