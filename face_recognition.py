# Import the libraries
import cv2
import time
# Loading the cascades
face_cascade = cv2.CascadeClassifier('/Users/Miranda/Documents/DeepLearning/ComputerVision/Module_1_Face_Recognition/haarcascade_frontalface_default.xml')
#eye_cascade = cv2.CascadeClassifier('/Users/Miranda/Documents/DeepLearning/ComputerVision/Module_1_Face_Recognition/haarcascade_eye.xml')
mask_cascade = cv2.CascadeClassifier('/Users/Miranda/Documents/DeepLearning/ComputerVision/Module_1_Face_Recognition/haarcascade_mcs_nose.xml')

# Adjust threshold value in range 80 to 105 based on your light.
bw_threshold = 80

# We are going to show some meesages to user. Define the attirbutes
font = cv2.FONT_HERSHEY_SIMPLEX
org = (30,30)
weared_mask_font_color = (255,255,255)
not_weared_mask_font_color = (0,0,255)
thickness = 2
font_scale = 1


# Function that will do the detection of the face and the eyes
# Input: Original Image (frame), B/W Image (gray)

def detect(gray, frame):
    # Get the coordinates of the rectangule that will get the face.
    # Create some tuples into a variable called faces. Tuples are 4 elements (X, Y which are 
    # the coordinates of upper left corner and the width & height of the rectangule)
    # To get these tuples we will use of the method detectMultiScale CascadeClassifier class
    # We pass 3 arguments. 1.3 & 5 are good values for Scale factor reduction and for minimun 
    # number of neighbor to accept the zone of pixels.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # Full loop that will iterate thorugh these faces and for each of them 
    # we will draw a rectangle  and will detect some eyes
    
    for (x,y,w,h) in faces:
        # 1) Draw the rectangule. This function will print the rectangule in the frame
        cv2.rectangle(frame,(x,y),(x+w, y+h),(255,0,0), 2)
        cv2.putText(frame, 'Cara detectada', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        # 2) Detect of the eyes: For that we need 2 region of interest
        # First region of interest for the B/W image
        roi_gray = gray[y:y+h, x:x+w]
        # First region of interest for the Original image
        roi_color = frame[y:y+h, x:x+w]
        #Detect the mask - We use roi_gray as mask are within the face
        mask = mask_cascade.detectMultiScale(roi_gray, 1.3, 7)
        for (mx,my,mw,mh) in mask:
            # 1) Draw the rectangule.This function will print the rectangule in the roi_color
            cv2.rectangle(roi_color,(mx,my),(mx+mw, my+mh),(0,255,0), 3)
            if (len(mask)==0):
                cv2.putText(frame, 'Con Mascarilla', (mw, mh-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 1)
            else:
                cv2.putText(frame, 'SIN MASCARILLA', (30, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
        #Detect the eyes - We use roi_gray as eyes are within the face
        #eyes = eye_cascade.detectMultiScale(roi_gray, 1.7, 25)
        #for (ex,ey,ew,eh) in eyes:
            # 1) Draw the rectangule.This function will print the rectangule in the roi_color
            #cv2.rectangle(roi_color,(ex,ey),(ex+ew, ey+eh),(0,255,0), 2)
    return frame    

# Doing Face recognition with the webcam
# We need the last frame coming from the webcam. We create an object from the 
# video capture class

video_capture = cv2.VideoCapture(0)
#time.sleep(20)
#ds_factor = 0.7
#Infinite loop until break
while True:
    # _ will ignore the first argument returned
    _, frame= video_capture.read()
    frame = cv2.flip(frame,1)
#    frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
    # Transform the original color frame to B/W by COLOR_BGR2GRAY
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canvas = detect(gray, frame)
    #Display all the sucessfull images in the window
    cv2.imshow('Video', canvas)
    # While exit - If we pray q we exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break    
# turn_off the webcan
video_capture.release()
cv2.destroyAllWindows()
    
    
    