import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Define the path for training images

path = 'faces'

images = []
classNames = []

# Reading the training images and classes and storing into the corresponsing lists
for img in os.listdir(path):
    image = cv2.imread(f'{path}/{img}')
    images.append(image)
    classNames.append(os.path.splitext(img)[0])
    print(classNames)


# Function for Find the encoded data of the imput image
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Find encodings of training images

encodeListKnown = findEncodings(images)
print('Encoding Complete')
 
# Define a videocapture object
cap = cv2.VideoCapture(0)
 
while True:
    success, img = cap.read()  # Reaading Each frame
    
    # Resizing the frame
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Finding the face location and endoings for the current frame
    
    face_locations = face_recognition.face_locations(imgS)
    face_encodes = face_recognition.face_encodings(imgS,face_locations)

    # Finding the matches for each detection
    
    for encodeFace,faceLoc in zip(face_encodes,face_locations):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)
        
        # If match found then get the classname for corresponding match

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()

        else:
            name = 'Unknown'

        # Draw the detection and names on the frame

        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),2)


    # show the output
    cv2.imshow('Webcam',img)

    if cv2.waitKey(1) == ord('q'):
        break

# release the camera object

cap.release()
cv2.destroyAllWindows()