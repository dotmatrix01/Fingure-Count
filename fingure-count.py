import cv2 as cv
import mediapipe as mp

# Initializing Hand Tracking Modules
mpHands = mp.solutions.hands
Hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingersCord=[(8,6),(12,10),(16,14),(20,18)]
thumbCord=(4,3)

# Capturing Video From Camera 
cap = cv.VideoCapture(0)

# Checking Camera is Opened or Not 
while (cap.isOpened()):
    count=0
    success , img = cap.read() # reading Frame 
    converted_image = cv.cvtColor(img,cv.COLOR_BGR2RGB) # converting BGR to RGB
    results = Hands.process(converted_image) # Processing Image for Tracking 
    HandNo=0
    lmlist=[]
    
    if results.multi_hand_landmarks: # Getting Landmark(location) of Hands if Exists 
        for id,lm in enumerate(results.multi_hand_landmarks[HandNo].landmark):
            h,w,c=img.shape
            cx,cy=int(lm.x*w),int(lm.y*h)
            lmlist.append((cx,cy))

        for hand_in_frame in results.multi_hand_landmarks: # looping through hands exists in the Frame 
            mpDraw.draw_landmarks(img,hand_in_frame, mpHands.HAND_CONNECTIONS) # drawing Hand Connections   
        for point in lmlist:
            cv.circle(img,point,5,(0,255,0),cv.FILLED)
        for coordinate in fingersCord:
            if lmlist[coordinate[0]][1] < lmlist[coordinate[1]][1]:
                count+=1
        if lmlist[thumbCord[0]][0] > lmlist[thumbCord[1]][0]:
            count+=1
        cv.putText(img,str(count),(175,175),cv.FONT_HERSHEY_PLAIN,10,(0,0,255),10)    
    cv.imshow("HandTracking", img) # showing Video 

    if cv.waitKey(1) == 113: # 113 - Q : press on Q : Close Video 
        break