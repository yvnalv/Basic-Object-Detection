#import the packages
from imutils.video import VideoStream
import datetime
import imutils
import time
import cv2

#type 'webcam' for using camera
#type 'filepath' to read from video file
src = 'webcam'
filepath = ''

#set minimum area that we allow for a motion detection
min_area = 1000

if src == 'webcam':
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

elif src == 'filepath':
    vs = cv2.VideoCapture(filepath)

else:
    print("Type 'webcam' for using camera, or 'filepath' to read from video file")

#use first frame as our background model
firstFrame = None

#loop over the frames of the video
while True:
    #grab the current frame and initialize whether it detects something
    frame = vs.read()
    frame = frame if src == 'webcam' else frame[1]
    text = 'No motion detected'

    #end the video if there is no frame left
    if frame is None:
        break

    #resize the frame, convert to grayscale and blur it
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    #if the first frame is None, initialize it
    if firstFrame is None:
        firstFrame = gray
        continue

    #compute the absolute difference between current frame and first frame
    frameDiff = cv2.absdiff(firstFrame, gray)
    thresh = cv2.threshold(frameDiff, 25, 255, cv2.THRESH_BINARY)[1]

    #dilate the thresholded image to fill in holes, then
    #find contours on thresholded image
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    #loop over the contours
    for c in cnts:
        #if the contours is too small, ignore it
        if cv2.contourArea(c) < min_area:
            continue

        
        #compute bounding box for the contour, draw it on
        #the frame and update the status
        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h),
                      (0, 255, 0), 2)
        text = "Motion Detected"

    #draw the text and timestamp on the frame
    cv2.putText(frame, 'Status : {}'.format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    #show the frame and record if the user press the key
    cv2.imshow('Object Status', frame)
    cv2.imshow('Thresh', thresh)
    cv2.imshow('Frame Diff', frameDiff)
    key = cv2.waitKey(1) & 0xFF

    #if the 'q' is pressed, quit
    if key == ord('q'):
        break

#cleanup camera and close any open windows
vs.stop() if src == 'webcame' else vs.release()
cv2.destroyAllWindows()

