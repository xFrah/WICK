from imutils.video import VideoStream

import argparse
import warnings
import datetime
import imutils
import json
import tflite_runtime.interpreter as tflite
import time
import cv2
import os
import numpy as np
from PIL import Image
from datetime import datetime
from uuid import uuid4
import serial
# import Servo

myservo = []
closed_position = 0
open_position = 180


def normalize(height, maximum, n):
    return (height * n) / maximum


def create_image(values, height, width, max_spectrometer_value):
    pixel_values = [((i + 1) * (255 // len(values))) for i in range(len(values))]
    #print(pixel_values)
    band_width = width // len(values)
    base = np.zeros((height, width))
    for i, row in enumerate(base):
        for y, (color, value) in enumerate(zip(pixel_values, values)):
            value = normalize(height, max_spectrometer_value, value)
            value *= 2 if y < 2 else 5
            row[band_width * y:band_width * (y + 1)] = [color if value > i else 0] * band_width
    return base


def parse_serial(stringa):
    try:
        #print(stringa)
        return [int(float(x)) for x in str(stringa).lstrip("b'").replace("\\r\\n'", "").split(", ") if x != ""]
    except ValueError:
        print("Error while parsing serial: " + str(stringa))
        return None


def setup_servos():
    for i in range(0, 16):
        myservo.append(Servo.Servo(i))
        Servo.Servo(i).setup()


def open_thing(servo_number):
    for i in range(closed_position, open_position, 5):
        myservo[servo_number].write(i)
        time.sleep(0.1)
    time.sleep(2)
    for i in range(closed_position, open_position, 5)[::-1]:
        myservo[servo_number].write(i)
        time.sleep(0.1)


# oh my god
# Load the TFLite model and allocate tensors.
interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open("conf.json"))
client = None

os.system("sudo chmod 666 /dev/ttymxc2")

vs = VideoStream(src=0).start()

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None

print("[INFO] Ready")

# loop over the frames of the video
while True:

    if input() != "q":
        continue
    print("[INFO] Taking snapshot")

    # grab the current frame and initialize the occupied/unoccupied
    # text
    original_frame = vs.read()
    # timestamp = datetime.datetime.now()
    #    text = "Unoccupied"

    # resize the frame, convert it to grayscale, and blur it
    # frame = imutils.resize(original_frame, width=500)

    #    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    #
    #    # if the average frame is None, initialize it
    #    if avg is None:
    #        print("[INFO] starting background model...")
    #        avg = gray.copy().astype("float")
    #        continue
    #
    #    # accumulate the weighted average between the current frame and
    #    # previous frames, then compute the difference between the current
    #    # frame and running average
    #    cv2.accumulateWeighted(gray, avg, 0.5)
    #    frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(avg))
    #
    #    # threshold the delta image, dilate the thresholded image to fill
    #    # in holes, then find contours on thresholded image
    #    thresh = cv2.threshold(frameDelta, conf["delta_thresh"], 255, cv2.THRESH_BINARY)[1]
    #    thresh = cv2.dilate(thresh, None, iterations=2)
    #    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #    cnts = imutils.grab_contours(cnts)

    # loop over the contours
    #    for c in cnts:
    #        # if the contour is too small, ignore it
    #        if cv2.contourArea(c) < conf["min_area"]:
    #            continue
    #
    #        #movement_detected = True
    #        #last_movement = datetime.datetime.now()
    #
    #        # compute the bounding box for the contour, draw it on the frame,
    #        # and update the text
    #        (x, y, w, h) = cv2.boundingRect(c)
    #        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #        text = "Occupied"
    #
    #    cv2.putText(frame, "Room Status: {}".format(text), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # check to see if the frames should be displayed to screen
    if conf["show_video"]:
        # display the security feed
        # show the frame and record if the user presses a key

        #        cv2.imshow("Thresh", thresh)
        #        cv2.imshow("Frame Delta", frameDelta)
        # cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        # if key == ord("q"):
        arduino = serial.Serial(port="/dev/ttymxc2", baudrate=9600)
        print("[INFO] Serial connection open")
        while True:
            while not arduino.is_open:
                pass
            raw = arduino.readline()
            print(raw)
            data = parse_serial(raw)
            if not data:
                continue
            if len(data) != 6:
                continue
            print("[INIT] " + str(data))
            data = parse_serial(arduino.readline())
            while len(data) != 6:
                data = parse_serial(arduino.readline())
            print("[CONFIRM] " + str(data))
            # uuid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            uuid = "prediction"
            im = Image.fromarray(original_frame)
            im.save("images/" + uuid + ".png")
            with open("images/" + uuid + ".txt", "w+") as f:
                f.write(str(data))
            image = create_image(data, len(original_frame), len(original_frame[0]), 1000)
            im = Image.fromarray(image)
            im = im.convert("L")
            im.save("images/" + uuid + "-[SPECTRO].png")
            # concatenate image Horizontally
            # horizontal = np.concatenate((original_frame, cv2.imread("images/" + uuid + "-[SPECTRO].png")), axis=1)
            # cv2.imshow("Image", horizontal)
            cv2.waitKey(500)
            arduino.close()
            if True:
                image = cv2.imread("images/" + uuid + ".png")
                spectro = cv2.imread("images/" + uuid + "-[SPECTRO].png", cv2.IMREAD_UNCHANGED)
                result = np.dstack((image, spectro))

                # Test the model on random input data.
                input_shape = input_details[0]['shape']
                # print(input_shape)
                # input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
                # r"C:\Users\fdimo\Desktop\coral_images\
                input_data = result
                input_data = input_data.astype(np.float32)
                # print(interpreter.get_input_details())
                input_data = np.expand_dims(input_data, axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)

                class_names = ['can', 'paper', 'plastic', 'tissues']
                interpreter.invoke()

                # The function `get_tensor()` returns a copy of the tensor data.
                # Use `tensor()` in order to get a pointer to the tensor.
                output_data = interpreter.get_tensor(output_details[0]['index'])
                print(output_data)
                print(class_names[np.argmax(output_data[0])])
            break
            # image = create_image(data, 255, 255, 1000)
            # im = Image.fromarray(image)
            # im = im.convert("L")
            # im.save("spectro.jpeg")
            # img = cv2.imread("spectro.jpeg")
            # cv2.imshow("Image", img)
