import cv2
import keyboard
from sys import maxsize
import cvzone

detector_fa=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
detector_sm=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_smile.xml")
detector_ey=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")

emoji_f=cv2.imread("faCE.png",cv2.IMREAD_UNCHANGED)
emoji_s=cv2.imread("lip.png",cv2.IMREAD_UNCHANGED)
emoji_e=cv2.imread("eye.png",cv2.IMREAD_UNCHANGED)

video_capture=cv2.VideoCapture("video.mp4")

while True:
    ret,frame=video_capture.read()
    if ret==False:
        break

    k=cv2.waitKey(1)
    if keyboard.is_pressed("1"):
        FACES=detector_fa.detectMultiScale(frame,1.3)

        for (x,y,w,h)in FACES:
            final_emoji=cv2.resize(emoji_f,(w,h))
            frame=cvzone.overlayPNG(frame,final_emoji,[x,y])

    if keyboard.is_pressed("2"):
        EYE=detector_ey.detectMultiScale(frame,2,maxsize=(50,50))

        for (x,y,w,h) in EYE:
            final_emoji=cv2.resize(emoji_e,(w,h))
            frame=cvzone.overlayPNG(frame,final_emoji,[x,y])
        
        LIP=detector_sm.detectMultiScale(frame,1.3)
        
        for (x,y,w,h) in LIP:
            final_emoji=cv2.resize(emoji_s,(w,h))
            frame=cvzone.overlayPNG(frame,final_emoji,[x,y])

    if keyboard.is_pressed("3"):
        FACE=detector_fa.detectMultiScale(frame,1.3)

        for (x,y,w,h) in FACE:
            blurred=frame[y:y+h,x:x+w]
            pixel=cv2.resize(blurred,(w,h),interpolation=cv2.INTER_LINEAR)
            output=cv2.resize(pixel,(w,h),interpolation=cv2.INTER_NEAREST)
            frame[y:y+h,x:x+w]=output

    if keyboard.is_pressed("4"):
        frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
        frame_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        frame_gray=255-frame_gray
        cv2.imshow("output",frame_gray)
        cv2.waitKey()
    
    if keyboard.is_pressed("5"):
        FACESS=detector_fa.detectMultiScale(frame,1.3)

        for (x,y,w,h) in FACESS:
            x,y,w,h=FACESS
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),8)

        cv2.imshow("output",frame)
        cv2.waitKey()

    if keyboard.is_pressed("Esc"):
        exit()

cv2.imshow("output",video_capture)
cv2.waitkey()