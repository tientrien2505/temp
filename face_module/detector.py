from abc import ABC, abstractmethod
from face_recognition import face_locations
from insightface.model_zoo import get_model

class Detector(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def detect(self, img):
        pass

class HogDetector(Detector):
    def __init__(self, *args, **kwargs):
        super(HogDetector, self).__init__()

    def detect(self, img):
        _face_locations = face_locations(img, model='hog')
        return _face_locations

class CnnDetector(Detector):
    def __init__(self, *args, **kwargs):
        super(CnnDetector, self).__init__()

    def detect(self, img):
        _face_locations = face_locations(img, model='cnn')
        return _face_locations


class RetinaFaceDetector(Detector):
    def __init__(self, model_name='retinaface_r50_v1', gpu=False):
        self._model = get_model(model_name)
        _ctx_id = 1 if gpu else -1
        self._model.prepare(ctx_id=_ctx_id, nms=0.4)
        super(RetinaFaceDetector, self).__init__()

    def detect(self, img):
        _face_locations, _ = self._model.detect(img, threshold=0.5, scale=1.0)
        _face_locations = _face_locations[:, [1, 2, 3, 0]]
        return _face_locations

class DetectorBuilder():
    detectors = {
        'cnn': CnnDetector,
        'hog': HogDetector,
        'retina': RetinaFaceDetector
    }
    def __init__(self):
        pass

    @classmethod
    def build(cls, detector_type='hog', *args, **kwargs):
        _detector = cls.detectors[detector_type]
        return _detector(*args, **kwargs)

if __name__ == '__main__':
    import cv2
    dt = DetectorBuilder.build('retina')
    img = cv2.imread('/home/truongdt/projects/faceid/son_face_core/test.jpg')
    det = dt.detect(img)
    for d in det:
        top, right, bottom, left = d
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
        img = cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 5)
    cv2.imwrite('abc.jpg', img)