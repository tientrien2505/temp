from face_recognition import face_locations

detectors = {
    'face_recognition': face_locations
}

class Detector():
    def __init__(self, detector='face_recognition'):
        self.detector = detectors[detector]

    def detect(self, img, **kwargs):
        """
        :param img:
        :param kwargs:
        :return: A list of tuples of found face locations in css (top, right, bottom, left) order
        """
        return self.detector(img, **kwargs)