import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model  # TensorFlow is required for Keras to work
from keras.layers import DepthwiseConv2D
import cv2  # Install opencv-python
import numpy as np

# Custom DepthwiseConv2D to remove 'groups' argument
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, **kwargs):
        kwargs.pop('groups', None)  # Remove the 'groups' argument if it's present
        super().__init__(**kwargs)

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model with the custom layer
model = load_model("model/keras_Model.h5", compile=False, custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load the labels
class_names = open("model/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on default camera of your computer
camera = cv2.VideoCapture(0)

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
        if previous_class_name == "biowaste":
            print("Bio waste found in trash.")
        elif previous_class_name == "plastic":
            print("Plastic waste found in trash.")
        elif previous_class_name == "metal":
            print("Metal waste found in trash.")
        else:
            print("I did'nt understand the waste category.")

    # Listen to the keyboard for presses.
    keyboard_input = cv2.waitKey(1) 

    # 27 is the ASCII for the esc key on your keyboard.
    if keyboard_input == 27:
        break

camera.release()
cv2.destroyAllWindows()
