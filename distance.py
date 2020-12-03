import numpy as np
import cv2 as cv

faceDetector = cv.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

def face_data(frame):

    width = 0

    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faceDetected = faceDetector.detectMultiScale(grayImage, 1.3, 5)

    for x, y, w, h in faceDetected:
        width = w
    return w




camera = cv.VideoCapture(0)

width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
size = (width, height)

fourcc = cv.VideoWriter_fourcc(*'MJPG')
out = cv.VideoWriter('saved/example.avi', fourcc, 20.0, size)

distance = 7.5
true_width = 14.3

example_image = cv.imread('images/example.jpg')

width_from_the_face = face_data(example_image)

focal_length = (width_from_the_face * distance)/ width_from_the_face


while camera.isOpened():

    sucess, frame = camera.read()

    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faceDetected = faceDetector.detectMultiScale(grayImage, 1.3, 5)

    for x, y, w, h in faceDetected:

        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

        if w != 0:

            new_distance = (focal_length * true_width)/w

            cv.putText(frame, f"Distancia da camera = {round(new_distance, 5)} m", (50, 30), cv.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)


    cv.imshow('face detected', cv.resize(frame, (480, 720)))

    if sucess == True:
        out.write(frame)


    if cv.waitKey(1) == ord('q'):
        break

camera.release()
out.release()
cv.destroyAllWindows()










