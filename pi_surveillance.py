from imutils.video import VideoStream

import argparse
import warnings
import datetime
import imutils
import json
import time
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
from uuid import uuid4
import serial


def normalize(height, maximum, n):
    return (height * n) / maximum


def create_image(values, height, width, max_spectrometer_value):
    pixel_values = [((i + 1) * (255 // len(values))) for i in range(len(values))]
    print(pixel_values)
    band_width = width // len(values)
    base = np.zeros((height, width))
    for i, row in enumerate(base):
        for y, (color, value) in enumerate(zip(pixel_values, values)):
            value = normalize(height, max_spectrometer_value, value)
            row[band_width * y:band_width * (y + 1)] = [color if value > i else 0] * band_width
    return base


def parse_serial(stringa):
    return [int(x) for x in str(stringa).lstrip("b'").replace("\\r\\n'", "").split(", ") if x != ""]


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--conf", required=True,
                help="path to the JSON configuration file")
args = vars(ap.parse_args())

# filter warnings, load the configuration and initialize the Dropbox
# client
warnings.filterwarnings("ignore")
conf = json.load(open(args["conf"]))
client = None

vs = VideoStream(src=0).start()

# allow the camera to warmup, then initialize the average frame, last
# uploaded timestamp, and frame motion counter
print("[INFO] warming up...")
time.sleep(conf["camera_warmup_time"])
avg = None

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1)

print("Ready")

# loop over the frames of the video
while True:

    if input() != "q":
        continue
    print("Taking snapshot")

    # grab the current frame and initialize the occupied/unoccupied
    # text
    original_frame = vs.read()
    # timestamp = datetime.datetime.now()
#    text = "Unoccupied"

    # resize the frame, convert it to grayscale, and blur it
    frame = imutils.resize(original_frame, width=500)

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
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key is pressed, break from the lop
        #if key == ord("q"):
        while True:
            data = parse_serial(arduino.readline())
            if len(data) != 8:
                continue
            uuid = datetime.now().strftime('%Y%m-%d%H-%M%S-') + str(uuid4())
            im = Image.fromarray(original_frame)
            im.save(uuid + ".png")
            with open(uuid + ".txt", "w+") as f:
                f.write(str(data))
            image = create_image(data, len(original_frame), len(original_frame[0]), 1000)
            im = Image.fromarray(image)
            im = im.convert("L")
            cv2.imshow("Image", im)
            im.save(uuid + "-[SPECTRO].png")
            cv2.waitKey(500)
            break
            #image = create_image(data, 255, 255, 1000)
            #im = Image.fromarray(image)
            #im = im.convert("L")
            #im.save("spectro.jpeg")
            #img = cv2.imread("spectro.jpeg")
            #cv2.imshow("Image", img)

