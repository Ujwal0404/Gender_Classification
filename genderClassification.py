import pickle
pickle_in=open('genderClassification.pickle','rb')
pickle_in2=open('pca.pickle','rb')
clf=pickle.load(pickle_in)
pca=pickle.load(pickle_in2)
import cv2
import numpy as np

face_1=[]
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture=cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = faceCascade.detectMultiScale(gray, 1.3, 5,minSize=(30,30))
    for i in range(len(detections)):
        face_i = detections[i]
        x, y, w, h = face_i
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 222, 0), 1)
        f=frame[y:y+h,x:x+w]
        f2=cv2.resize(f,(90,90),interpolation=cv2.INTER_AREA)
        f3=cv2.cvtColor(f2,cv2.COLOR_BGR2GRAY)
        test = pca.transform(f3.reshape(1, -1))
        #print(test.shape)
        y=clf.predict(test)
        #face_1.append(f3)
        #cv2.imwrite('picture4.jpg',f)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if y==0:
            cv2.putText(frame, "Female", (x, y + h - 5), font, 2, (217, 243, 255), 2, cv2.LINE_AA)

        if y==1:
            cv2.putText(frame,"Male",(x,y+h-5), font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    if(cv2.waitKey(1)==ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
#cv2.imwrite('picture5.jpg',face_1[0])
#cv2.imwrite('picture6.jpg',face_1[1])
#print(face_1[0].shape)
#print(face_1[1].shape)
#g=cv2.imread('picture5.jpg',0)
#test=pca.transform(g.reshape(1,-1))
#print(test.shape)
#print(clf.predict(test))