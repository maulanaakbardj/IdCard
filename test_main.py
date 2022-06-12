import cv2
import matplotlib.pyplot as plt
import face_recognition

img1=face_recognition.load_image_file('0001.jpg')
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
locate_face1=face_recognition.face_locations(img1)[0]
encode_face1=face_recognition.face_encodings(img1)[0]
box_face1=cv2.rectangle(img1,(locate_face1[3],locate_face1[0]),(locate_face1[1],locate_face1[2]),(255,0,0),2)
cv2.imshow('0001.jpg', box_face1)
cv2.waitKey(0)
