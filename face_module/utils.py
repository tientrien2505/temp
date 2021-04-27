import cv2
import numpy as np
import dlib
from imutils.face_utils import FaceAligner

class FaceStraighten:
    def __init__(self, face_size=224):
        predictor_path = 'app/static/models/shape_predictor_68_face_landmarks.dat'
        predictor = dlib.shape_predictor(predictor_path)
        self.fa = FaceAligner(predictor, desiredFaceWidth=224)   

    def align(self, image, bb, padding=45):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # left, top, right, bottom = bb
        rect = dlib.rectangle(int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        face_aligned = self.fa.align(image, gray, rect)
        face_aligned = face_aligned[padding: 224-padding, padding: 224-padding]
        return face_aligned


def get_centers(landmarks):
    EYE_LEFT_OUTTER = landmarks[2]
    EYE_LEFT_INNER = landmarks[3]
    EYE_RIGHT_OUTTER = landmarks[0]
    EYE_RIGHT_INNER = landmarks[1]

    x = ((landmarks[0:4]).T)[0]
    y = ((landmarks[0:4]).T)[1]
    A = np.vstack([x, np.ones(len(x))]).T
    k, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    x_left = (EYE_LEFT_OUTTER[0]+EYE_LEFT_INNER[0])/2
    x_right = (EYE_RIGHT_OUTTER[0]+EYE_RIGHT_INNER[0])/2
    LEFT_EYE_CENTER =  np.array([np.int32(x_left), np.int32(x_left*k+b)])
    RIGHT_EYE_CENTER =  np.array([np.int32(x_right), np.int32(x_right*k+b)])
    
    pts = np.vstack((LEFT_EYE_CENTER,RIGHT_EYE_CENTER))
    
    return LEFT_EYE_CENTER, RIGHT_EYE_CENTER

def landmarks_to_np(landmarks, dtype="int"):
    num = landmarks.num_parts
    
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((num, 2), dtype=dtype)
    
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, num):
        coords[i] = (landmarks.part(i).x, landmarks.part(i).y)
    # return the list of (x, y)-coordinates
    return coords

def get_aligned_face(gray, face_location, landmarks_predictor):

    rect = dlib.rectangle(face_location[0], face_location[1], face_location[2], face_location[3])
    landmarks = landmarks_predictor(gray, rect)
    landmarks = landmarks_to_np(landmarks)

    LEFT_EYE_CENTER, RIGHT_EYE_CENTER = get_centers(landmarks)
    left = LEFT_EYE_CENTER
    right = RIGHT_EYE_CENTER

    desired_w = 256
    desired_h = 256
    desired_dist = desired_w * 0.5
    
    eyescenter = ((left[0]+right[0])*0.5 , (left[1]+right[1])*0.5)# 眉心
    dx = right[0] - left[0]
    dy = right[1] - left[1]
    dist = np.sqrt(dx*dx + dy*dy)# 瞳距
    scale = desired_dist / dist # 缩放比例
    angle = np.degrees(np.arctan2(dy,dx)) # 旋转角度
    M = cv2.getRotationMatrix2D(eyescenter,angle,scale)# 计算旋转矩阵

    # update the translation component of the matrix
    tX = desired_w * 0.5
    tY = desired_h * 0.5
    M[0, 2] += (tX - eyescenter[0])
    M[1, 2] += (tY - eyescenter[1])

    aligned_face = cv2.warpAffine(gray,M,(desired_w,desired_h))
    
    return aligned_face


def judge_eyeglass(face_img, threshold):
    face_img = cv2.GaussianBlur(face_img, (11,11), 0)

    sobel_y = cv2.Sobel(face_img, cv2.CV_64F, 0 ,1 , ksize=-1) 
    sobel_y = cv2.convertScaleAbs(sobel_y) 
    # cv2.imshow('sobel_y',sobel_y)

    edgeness = sobel_y 
    
    retVal,thresh = cv2.threshold(edgeness,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    d = len(thresh) * 0.5
    x = np.int32(d * 6/7)
    y = np.int32(d * 3/4)
    w = np.int32(d * 2/7)
    h = np.int32(d * 2/4)

    x_2_1 = np.int32(d * 1/4)
    x_2_2 = np.int32(d * 5/4)
    w_2 = np.int32(d * 1/2)
    y_2 = np.int32(d * 8/7)
    h_2 = np.int32(d * 1/2)
    
    roi_1 = thresh[y:y+h, x:x+w] 
    roi_2_1 = thresh[y_2:y_2+h_2, x_2_1:x_2_1+w_2]
    roi_2_2 = thresh[y_2:y_2+h_2, x_2_2:x_2_2+w_2]
    roi_2 = np.hstack([roi_2_1,roi_2_2])
    
    measure_1 = sum(sum(roi_1/255)) / (np.shape(roi_1)[0] * np.shape(roi_1)[1])
    measure_2 = sum(sum(roi_2/255)) / (np.shape(roi_2)[0] * np.shape(roi_2)[1])
    measure = measure_1*0.3 + measure_2*0.7
    
    # cv2.imshow('roi_1',roi_1)
    # cv2.imshow('roi_2',roi_2)
    print(measure)
    
    if measure > threshold:
        judge = True
    else:
        judge = False
    print(judge)
    return judge

def detect_glasses(img, face_location, landmarks_predictor, threshold):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = get_aligned_face(gray, face_location, landmarks_predictor)
    having_glasses = judge_eyeglass(face, threshold)
    return having_glasses


