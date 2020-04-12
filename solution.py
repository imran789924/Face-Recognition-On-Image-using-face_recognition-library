# -*- coding: utf-8 -*-

import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "personal_data/train"
UNKNOWN_FACES_DIR = "personal_data/val"
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "hog"

print("Loading known faces")

known_faces = []
known_names = []


for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        print(filename)
        path = os.path.join(KNOWN_FACES_DIR, name, filename)
        image = face_recognition.load_image_file(f"{path}")
        encodings = face_recognition.face_encodings(image)
        if len(encodings) > 0:
            encoding = face_recognition.face_encodings(image)[0]
        else:
           print("No faces found in the image!")
           continue
        known_faces.append(encoding)
        known_names.append(name)






print("processing unknown faces")
for filename in os.listdir(UNKNOWN_FACES_DIR):
    print(filename)
    path = os.path.join(UNKNOWN_FACES_DIR, filename)
    print(path)
    image = face_recognition.load_image_file(f"{path}")
    
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image,locations)
   
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    
    for face_encoding, face_location in zip(encodings, locations):
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            
            top_left = (face_location[3], face_location[0])
            bottom_right = (face_location[1], face_location[2])
            
            color = [0,255,0]
            cv2.rectangle(image, top_left, bottom_right, color, FRAME_THICKNESS)
            
            top_left = (face_location[3], face_location[2])
            bottom_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)
            cv2.putText(image, match, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
            
    cv2.imshow(filename, image)
    cv2.waitKey(1000)
    #cv2.destroyWindow(filename)
            
    
    
    
    
    
    
