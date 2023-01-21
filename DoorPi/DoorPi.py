"""
uvicorn DoorPi:app --host 0.0.0.0 --port 8000 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem 
it works hehehehe wueweuweuuwe
"""
import cv2

import requests
from pydantic import BaseModel
from fastapi import FastAPI


app = FastAPI(ssl_keyfile="key.pem", ssl_certfile="cert.pem")

class Person(BaseModel):
    name: str
    isRegister: int

from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
# face verification with the VGGFace2 model

""" Old code for face extraction but now it's directly done with the haar cascade and much faster (10sec faster)

# Importing Image module from PIL package
import PIL
from PIL import Image as im
from numpy import asarray
from mtcnn.mtcnn import MTCNN
# extract a single face from a given photograph
def extract_face(img, required_size=(224, 224)):
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(img)
    print("extract face")
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    image = im.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    print("face extracted")
    return face_array

    def saveImage(img,type):
    img = im.fromarray(img)
    path = './resource'

    if not os.path.exists(path):
        os.makedirs(path)

    filename = 'expected_face' + '.' + type.split('/')[-1]

    img.save('./resource' + '/' + filename)
    return './resource' + '/' + filename
        
"""

def get_webcam_face():
    face_cascade = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')
    img = None
    faceStrike = 0
    print("video capture")
    camera = PiCamera()
    # Set the camera resolution
    camera.resolution = (640, 480)
    # Set the number of frames per second
    camera.framerate = 32
    # Generates a 3D RGB array and stores it in rawCapture
    raw_capture = PiRGBArray(camera, size=(640, 480))
    # Wait a certain number of seconds to allow the camera time to warmup
    time.sleep(0.1)
    countingZeros = 0
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        face_detection=face_cascade.detectMultiScale(frame.array, scaleFactor=1.1, minNeighbors=2)
        print(faceStrike)
        if (len(face_detection) > 0):
            faceStrike = faceStrike + 1
            countingZeros = 0
        else:
            faceStrike = 0
            countingZeros = countingZeros + 1
        if (faceStrike > 10):
            img = frame.array
            for (x,y,w,h) in face_detection:
                img = img[y:y+h, x:x+w]
            break
        raw_capture.truncate(0)
        if (countingZeros > 30):
            print("Timeout for face detection")
            return None
    camera.close()
    print("release")
    return img
print("ok")


@app.post('/faceExtract')
def faceExtract(person: Person):
    name = person.name
    isRegistration = person.isRegister
    print("Starting face scan")
    #actual_face = extract_face(get_webcam_face()) # LESS GO THE THING BELOW IS WAY FASTER
    actual_face = get_webcam_face()
    if (actual_face is None):
        return "Timeout : no face detected"
    actual_face = cv2.resize(actual_face, (224, 224))
    #send the image over https to server
    #transform the actual_face numpy array to a python list
    actual_face = actual_face.tolist()
    if (isRegistration == 1):
        r = requests.post('https://rasp-manager:8000/saveFace', json={'imageArray': actual_face, 'name':name}, verify="/usr/share/ca-certificates/cert.pem")
    else:
        r = requests.post('https://rasp-manager:8000/faceVerification', json={'imageArray':actual_face, 'name':name}, verify="/usr/share/ca-certificates/cert.pem")
    print(r.text)
    return r.text

