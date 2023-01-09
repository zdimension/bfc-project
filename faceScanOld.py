import os
import cv2

# Importing Image module from PIL package
import PIL
from PIL import Image as im

# face verification with the VGGFace2 model
from numpy import asarray
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN

# Tensorflow version == 2.0.0
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dense,Dropout,Softmax,Flatten,Activation,BatchNormalization
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import tensorflow.keras.backend as K
def vggface():
    # Define VGG_FACE_MODEL architecture
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))	
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    # Load VGG Face model weights
    model.load_weights('vgg_face_weights.h5')
    return model

# extract a single face from a given photograph
def extract_face(img, required_size=(224, 224)):
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(img)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = img[y1:y2, x1:x2]
    # resize pixels to the model size
    image = im.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array
 
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(img):
    # extract faces
    test = []
    test.append(img)
    faces = [extract_face(f) for f in test]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    
    model = vggface() #VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat
 
# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    return score <= thresh
    
def compare_faces(face1, face2):
    embeddings1 = get_embeddings(face1)
    embeddings2 = get_embeddings(face2)
    return is_match(embeddings1, embeddings2)

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
    cap = cv2.VideoCapture()
    # The device number might be 0 or 1 depending on the device and the webcam
    cap.open(0, cv2.CAP_DSHOW)
    while(True):
        ret, frame = cap.read()
        face_detection=face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=2)
        if (len(face_detection) > 0):
            faceStrike = faceStrike + 1
        else:
            faceStrike = 0
        if (faceStrike > 10):
            img = frame
            break
    cap.release()
    return img


def main():

    expected_face = im.open("./resource/expected_face.jpg")   
    print("please stand in front of the camera for a face control.")
    actual_face = get_webcam_face()
    expected_face= asarray(expected_face)
    result = compare_faces(actual_face, expected_face)
    
    if result:
        print('Passenger validated, please proceed through gate')
    else:
        print('Passenger not validated, please wait for the security staff')

main()
