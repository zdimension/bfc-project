"""
uvicorn piserver:app --host 0.0.0.0 --port 8000 --ssl-keyfile ./key.pem --ssl-certfile ./cert.pem 
it works hehehehe wueweuweuuwe
"""
import os
import requests
from typing import Union
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile

app = FastAPI(ssl_keyfile="key.pem", ssl_certfile="cert.pem")

class Image(BaseModel):
    imageArray: list = []
    name : str

class Nfc(BaseModel):
    personTag: str

class Person(BaseModel):
    name: str
    isRegister: int

class DoorPi(BaseModel):
    ipAddress : str

doorPiIP = None

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input
from scipy.spatial.distance import cosine

# Load VGG Face model weights
model = tf.keras.models.load_model('vggface')
    
# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(img):

    # convert into an array of samples
    
    #samples = np.asarray(img, 'float32')
    img = np.expand_dims(img, axis=0)
    
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(img)
    # perform prediction
    yhat = np.squeeze(model.predict(samples), axis=0)
    return yhat
 
# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    return score <= thresh
    
def compare_faces(face1, face2):
    print("Get first face embedding from DoorPi")
    embeddings1 = get_embeddings(face1)
    print("Get second face embedding from saved vector image")
    embeddings2 = get_embeddings(face2)
    return is_match(embeddings1, embeddings2)

@app.post('/NfcVerification')
def NfcVerification(nfc: Nfc):
    print("Nfc verification started")
    personTag = nfc.personTag
    print("PersonTag received")
    # read a file line by line in /resource/ and check if personTag is present
    with open("./resource/tags.txt", "r") as f:
        for line in f:
            val = line.strip(' ')
            if personTag == val[1]:
                print("PersonTag verified")
                requests.post("https://DoorPi:8000/faceExtract", json = {"name":val[0], "isRegister":0}, verify="/usr/share/ca-certificates/cert.pem")
                return "Nfc verification completed"
    return "PersonTag not verified"
    
@app.post('/faceVerification')
def faceVerification(image: Image):
    print("Face verification started")
    img = np.asarray(image.imageArray)
    name = image.name
    print("Image and person received")
    expected_face = np.load("./resource/"+name+".npy")
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
    requests.post("https://DoorPi:8000/faceExtract", json = {"name": person.name,"isRegister": person.isRegister}, verify="/usr/share/ca-certificates/cert.pem")
    return "ok"


@app.post('/saveFace')
def saveFace(image: Image):
    print("Saving new registered face ") 
    img = np.asarray(image.imageArray)
    name = image.name
    with open("./resource/"+name+'.npy', 'wb') as f:
        np.save(f, img)
    return "ok"

"""@app.post('/registerNFC') # TODO probably not like this 
def registerFace(person: Person):
    print("NFC registration started") 
    print("person received")
    requests.post("https://DoorPi:8000/registerNFC", data = person) # TODO add registerNFC function to the server
    return "ok"
"""

@app.post('/init')
def initilize(doorPi: DoorPi):
    print("Initilization started")
    global doorPiIP
    doorPiIP = doorPi.ipAddress
    os.system("sudo python ./updateHosts.py "+doorPiIP)
    return "Ip address received and saved"

""" no longer necessary
@app.get('/initIP')
def test():
    return doorPiIP
"""
