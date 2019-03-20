import cv2
import numpy

cap = cv2.VideoCapture(0)

i = 0
while (True):
    i += 1
    ret, frame = cap.read()
    cv2.imshow("KIR", frame)

    p = cv2.waitKey(0) & 0xff

    if p == ord('q'):
        break
    elif p == ord('s'):
        cv2.imwrite('./saves/{}.JPG'.format(i), frame)
    else:
        continue
