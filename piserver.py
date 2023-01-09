import os
import flask 
import requests

app = flask.Flask(__name__)

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

@app.route('/NfcVerification', methods=['POST'])
def NfcVerification():
    print("Nfc verification started")
    personTag = request.args['personTag']
    print("PersonTag received")
    # read a file line by line in /resource/ and check if personTag is present
    with open("./resource/tags.txt", "r") as f:
        for line in f:
            val = line.strip(' ')
            if personTag == val[1]:
                print("PersonTag verified")
                requests.post("http://192.168.2.2/faceExtract", data = {'person':val[0], 'register': 0})
                break
        print("PersonTag not verified") 
    return "Nfc verification completed"

@app.route('/faceVerification', methods=['POST'])
def faceVerification():
    print("Face verification started")
    img = request.args['file'] 
    person = request.args['person']
    print("Image and person received")
    img = img_to_array(img)
    expected_face = np.array(im.open("./resource/"+person+".txt"))  # TODO change to .txt with an array of values this way we don't store the face
    result = compare_faces(img,expected_face)
    if result:
        print("Face verified")
        requests.post("http://192.168.2.3/doorUnlock")
    else:
        print("Face not verified")

@app.route('/registerFace', methods=['POST'])
def registerFace():
    print("Face registration started") 
    person = request.args['person']
    print("person received")
    requests.post("http://192.168.2.2/faceExtract", data = {'person':person, 'register': 1})

@app.route('/saveFace', methods=['POST'])
def saveFace():
    print("Saving new registered face ") 
    person = request.args['person']
    print("person received")
    requests.post("http://192.168.2.2/faceExtract", data = {'person':person, 'register': 1})
    img = request.args['file'] 
    person = request.args['person']
    print("Image and person received")
    img = img_to_array(img) # replace with numpy array function
    with open("./resource/"+person+'.npy', 'wb') as f:
        np.save(f, img)

@app.route('/registerNFC', methods=['POST']) # TODO probably not like this but needs to be checked 
def registerFace():
    print("NFC registration started") 
    person = request.args['person']
    print("person received")
    requests.post("http://192.168.2.2/", data = {'person':person, 'register': 1}) # TODO add registerNFC function to the server^l 
if __name__ == '__main__':
    app.run(host='192.168.2.1', port=5000, debug=True)