import os
import serial
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model  # TensorFlow is required for Keras to work
from keras.layers import DepthwiseConv2D
import cv2  # Install opencv-python
import numpy as np

# Set up serial communication (adjust 'COM3' to the correct port)
ser = serial.Serial('COM5', 9600)  # Change 'COM3' to the port your Arduino is connected to
time.sleep(2)  # Wait for the connection to establish

# Custom DepthwiseConv2D to remove 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument if it's present
        super().__init__(**kwargs)

def send_angle(servo, angle):
    """Send the angle for a specified servo."""
    command = f"S{servo}:{angle}\n"
    ser.write(command.encode())  # Send the command to Arduino
    time.sleep(0.1)  # Small delay for stability

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model with the custom layer
model = load_model("model/keras_Model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(1)

if not camera.isOpened():
    print("Failed to open the camera.")
else:
    print("Camera opened successfully.")

# Variable to store the previous class name
previous_class_name = None

while True:
    # Grab the webcamera's image.
    ret, image = camera.read()

    # Resize the raw image into (224-height,224-width) pixels
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape.
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image = (image / 127.5) - 1

    # Predict the class with model and suppress the progress bar output
    prediction = model.predict(image, verbose=0)  # Set verbose to 0 to suppress output
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Strip to remove newline characters
    confidence_score = prediction[0][index]

    # Check if the confidence is above 90% and the class name has changed
    if confidence_score > 0.90 and class_name != previous_class_name:
        print(f"Class: {class_name} - Confidence Score: {np.round(confidence_score * 100, 2)}%")
        previous_class_name = class_name  # Update the previous class name
        # Use 'in' to check the class name
        if "biowaste" in class_name:
            print("Bio waste found in trash.")
            send_angle(2, 200)
            time.sleep(4)
            send_angle(1, 39)
            time.sleep(4)
            send_angle(1, 180)
            time.sleep(2)
            send_angle(2, 120)
            time.sleep(2)
        elif "plastic" in class_name:
            print("Plastic waste found in trash.")
            send_angle(2, 50)
            time.sleep(4)
            send_angle(1, 39)
            time.sleep(4)
            send_angle(1, 180)
            time.sleep(2)
            send_angle(2, 120)
            time.sleep(2)
        else:
            print("Unclassified waste detected.")
            send_angle(1, 180)
            send_angle(2, 120)
            time.sleep(2)


    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1) 

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()