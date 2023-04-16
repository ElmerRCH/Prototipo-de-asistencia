import cv2

capture = cv2.VideoCapture(0)

while (capture.isOpened()):
    ret, frame = capture.read()
    cv2.imshow('webCam',frame)
    capture.release()
    print("camara funcional")
    if (cv2.waitKey(6) == ord('s')):
        break

capture.release()
cv2.destroyAllWindows()