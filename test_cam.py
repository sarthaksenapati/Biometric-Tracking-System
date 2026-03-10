import cv2
cap = cv2.VideoCapture("http://192.168.0.183:4747/video")
print("Opened:", cap.isOpened())
ret, frame = cap.read()
print("Read:", ret)
cap.release()