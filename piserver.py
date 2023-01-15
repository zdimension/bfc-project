import os
import requests
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI

app = FastAPI(ssl_keyfile="key.pem", ssl_certfile="cert.pem")

class Image(BaseModel):
    file: Union[str, bytes]
    person: str

class Nfc(BaseModel):
    personTag: str

class Person(BaseModel):
    name: str
    register: int

class DoorPi(BaseModel):
    ipAddress : str

doorPiIP = None

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D,Convolution2D,MaxPooling2D
from tensorflow.keras.layers import Dropout,Flatten,Activation
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from scipy.spatial.distance import cosine
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

@app.post('/NfcVerification')
def NfcVerification(nfc: Nfc, person: Person):
    print("Nfc verification started")
    personTag = nfc.personTag
    print("PersonTag received")
    # read a file line by line in /resource/ and check if personTag is present
    with open("./resource/tags.txt", "r") as f:
        for line in f:
            val = line.strip(' ')
            if personTag == val[1]:
                print("PersonTag verified")
                requests.post(doorPiIP+"/faceExtract", data = person)
                break
        print("PersonTag not verified") 
    return "Nfc verification completed"

@app.post('/faceVerification')
def faceVerification(image: Image):
    print("Face verification started")
    img = image.file
    person = image.person
    print("Image and person received")
    img = img_to_array(img)
    expected_face = np.load("./resource/"+person+".npy")
    result = compare_faces(img,expected_face)
    if result:
        print("Face verified")
        #requests.post("http://192.168.2.3/doorUnlock")
        # TODO Replace with bluetooth unlock module
    else:
        print("Face not verified")

@app.post('/registerFace')
def registerFace(person: Person):
    print("Face registration started") 
    print("person received")
    requests.post(doorPiIP+"/faceExtract", data = person)
    return "ok"

@app.post('/saveFace')
def saveFace(image: Image):
    print("Saving new registered face ") 
    img = image.file
    person = image.person
    print("Image and person received")
    img = np.array(img) # TODO img probably not on the right var format since I'm taking it from a request 
    with open("./resource/"+person+'.npy', 'wb') as f:
        np.save(f, img)
    return "ok"
"""
@app.post('/registerNFC') # TODO probably not like this 
def registerFace(person: Person):
    print("NFC registration started") 
    print("person received")
    requests.post(doorPiIP"/registerNFC", data = person) # TODO add registerNFC function to the server
    return "ok"
"""


@app.post('/initilize')
def initilize(doorPi: DoorPi):
    print("Initilization started")
    doorPiIP = doorPi.ipAddress
    return "Ip address received and saved"