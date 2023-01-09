import os
import cv2

# Importing Image module from PIL package
import flask
import requests
import PIL
from PIL import Image as im
from picamera.array import PiRGBArray # Generates a 3D RGB array
from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
# face verification with the VGGFace2 model
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
    path = './resource'

    if not os.path.exists(path):
        os.makedirs(path)

    filename = 'expected_face' + '.' + type.split('/')[-1]
    with open(os.path.join(path, filename), 'wb') as temp_file:
        temp_file.write(img)
        return './resource' + '/' + filename
        
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
    for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
        face_detection=face_cascade.detectMultiScale(frame.array, scaleFactor=1.1, minNeighbors=2)
        print(faceStrike)
        if (len(face_detection) > 0):
            faceStrike = faceStrike + 1
        else:
            faceStrike = 0
        if (faceStrike > 10):
            img = frame.array
            break
        raw_capture.truncate(0)
    #cap.release()
    print("release")
    return img

# add an argument to main which contains a string of a person name
@app.route('/faceExtract', methods=['POST'])
def faceExtract():
    name = request.form['person'] 
    isRegistration = request.form['register']   
    print("Starting face scan")
    actual_face = get_webcam_face()
    # save image
    filename = saveImage(actual_face,'jpg')
    #send the image over https to server 192.168.102.1
    with open(filename, 'rb') as f:
        if (isRegistration == 1):
            r = requests.post('https://192.168.2.1/saveFace', files={'file': f, 'person':name})
        else:
            r = requests.post('https://192.168.2.1/faceVerification', files={'file': f, 'person':name}) #todo : change the ip address to the server ip address
    print(r.text)

if __name__ == '__main__':
    app.run(host='192.168.2.2', port=5000, debug=True)
    
