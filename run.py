import cv2
from face_module.recognizer import FaceRecognizer
from face_module.detector import Detector
recognizer = FaceRecognizer()
detector = Detector()
vc = cv2.VideoCapture(0)
while True:
    ret, img = vc.read()
    locs = detector.detect(img)
    if len(locs) > 0:
        top, right, bottom, left = locs[0]
        img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 5)
        face_img = img[top:bottom, left:right]
        label, prop = recognizer.recognize(face_img)
        img = cv2.putText(img, str(label) + f'accuracy: {prop:.2}', (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5)
    cv2.imshow('video', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

vc.release()
cv2.destroyAllWindows()