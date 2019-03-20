from cv2 import aruco
import cv2

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)

for i in range(10):
    cv2.imwrite("Markers/Marker_{:02}.png".format(i), aruco.drawMarker(aruco_dict, i, 200))
