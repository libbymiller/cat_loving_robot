#!/usr/bin/python

# camera
import cv2
import imutils
from imutils.video.pivideostream import PiVideoStream
from imutils.video import FPS
from picamera.array import PiRGBArray
from picamera import PiCamera

import os
import sys

import numpy as np

import time
import datetime


dir = "results"
if not os.path.exists(dir):
    os.makedirs(dir)

# cam
print("[INFO] cam sampling THREADED frames from `picamera` module...")
vs = PiVideoStream().start()
time.sleep(2.0)
fps = FPS().start()

def do_something_on_image(fn='/home/pi/camlive.jpg'):
    print(fn)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        cv2.imwrite(fn, frame)

        print('Captured %dx%d image' % ( frame.shape[1], frame.shape[0]) )
        time.sleep(2)
        dt = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        os.system("cp "+fn+" results/"+dt+".jpg")



if __name__ == '__main__':
  while True:
    try:
      do_something_on_image()
    except KeyboardInterrupt:
        print("Bye")
        sys.exit()
