import tensorflow as tf
import cv2
from face_recognition import face_locations
import numpy as np

img = cv2.imread('test.jpg')
locs = face_locations(img)
loc = None if len(locs) < 1 else locs[0]
top, right, bottom, left = loc
face_img = img[top:bottom, left:right]
print(face_img.shape)
interpreter = tf.lite.Interpreter(model_path='model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
print(input_details)
output_details = interpreter.get_output_details()
print(output_details)
input_shape = input_details[0]['shape']
face_img = cv2.resize(face_img, (224, 224))
face_img = face_img.astype('float32')
print(face_img.shape, face_img.dtype)
import time
start = time.time()
interpreter.set_tensor(input_details[0]['index'], np.expand_dims(face_img, 0))
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
end = time.time()
print('time: ', end-start)
print(output_data)
print(output_data.shape)
