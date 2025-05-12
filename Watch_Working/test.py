import serial
import time

# Simulated dataset
simulated_data = [
    [0, 85.47, 37.31, 12.8247, 80.0458],
    [0, 85.47, 37.31, 12.8247, 80.0458],
    [0, 85.47, 37.50, 12.8247, 80.0458],
    [0, 61.54, 37.31, 12.8247, 80.0458],
    [0, 61.54, 37.19, 12.8247, 80.0458],
    [0, 61.54, 37.25, 12.8247, 80.0458],
    [0, 61.54, 37.38, 12.8247, 80.0458],
    [0, 61.54, 37.31, 12.8247, 80.0458],
    [0, 61.54, 37.44, 12.8247, 80.0458],
    [0, 61.54, 37.31, 12.8247, 80.0458],
    [0, 85.47, 37.38, 12.8247, 80.0458],
    [0, 85.47, 37.44, 12.8247, 80.0458],
    [0, 85.47, 37.19, 12.8247, 80.0458],
    [0, 85.47, 37.38, 12.8247, 80.0458],
    [0, 85.47, 37.19, 12.8247, 80.0458],
    [0, 85.47, 37.31, 12.8247, 80.0458],
    [0, 85.47, 37.38, 12.8247, 80.0458],
]

# Change COM port to the paired port connected to your dashboard's COM12
try:
    ser = serial.Serial('COM11', 9600)  # This writes to COM11, paired to COM12
    print("Streaming simulated data to COM11 (connected to COM12)... Press Ctrl+C to stop.")
    
    i = 0
    while True:
        data = simulated_data[i % len(simulated_data)]
        formatted = "[" + ", ".join(map(str, data)) + "]\n"
        ser.write(formatted.encode('utf-8'))
        time.sleep(1)  # Simulate 1-second sensor data interval
        i += 1

except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
