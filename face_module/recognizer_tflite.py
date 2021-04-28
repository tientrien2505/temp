import tensorflow as tf
import numpy as np

class FaceRecognizerTflite():
    def __init__(self, model_path='model/model.tflite'):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']

    def get_embedding(self, face_img):
        face_img = face_img.astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], np.expand_dims(face_img, 0))
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])

        return output_data

