# Installing dependencies
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Declaring global variables
path = 'ImagesAttendance'
images =[]
class_names = []
name_list = os.listdir(path)

print(name_list)

# Initiating an iteration to read and store images with their names
for cl in name_list:

    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    class_names.append(os.path.splitext(cl)[0])

print(class_names)

# Encoding all the images
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)

    return encode_list

# Giving attendance inputs to the csv file
def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        data = f.readlines()
        print(data)
        
        nl =[]

        for line in data:
            entry = line.split(',')
            nl.append(entry[0])
        
        if name not in nl:
            now = datetime.now()
            dt_str = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dt_str}')

encode_known = find_encodings(images)
print('Encoding Complete')


# Integrating camera with the program to take inputs in Real-Time
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    s_img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    s_img = cv2.cvtColor(s_img,  cv2.COLOR_BGR2RGB)

# Encoding inputs from the camera
    face_cur_frame = face_recognition.face_locations(s_img)
    encode_cur_frame = face_recognition.face_encodings(s_img, face_cur_frame)

# Matching the captured faces with the database
    for encode_face, face_loc in zip(encode_cur_frame, face_cur_frame):
        matches = face_recognition.compare_faces(encode_known, encode_face)
        face_dis = face_recognition.face_distance(encode_known, encode_face)
        print(face_dis)

        match_index = np.argmin(face_dis)

# Highlighting the faces detected and showing their respective names by plotting a block around it
        if matches[match_index]:
            name = class_names[match_index].upper()
            print(name)

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 253, 208), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (255, 253, 208), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 139), 2)
            attendance(name)

# Deploying the camera  
    cv2.imshow('Webcam', img)
    cv2.waitKey(1)