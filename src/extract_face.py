import cv2
import torch
import numpy as np
from PIL import Image
import face_alignment
from skimage.transform import SimilarityTransform


# Target facial landmarks used for similarity transformation
REF_LEFT_EYE  = (0.25, 0.4)
REF_RIGHT_EYE = (0.75, 0.4)
REF_NOSE      = (0.5,  0.5)
# Face landmark indices
NOSE_IDX      = 31
LEFT_EYE_IDX  = 37
RIGHT_EYE_IDX = 43


def alignment(src_img, landmarks, target_width, target_height):
    ref_pts = [REF_LEFT_EYE, REF_RIGHT_EYE, REF_NOSE]
    # Scale ref poins to actual image size
    ref_pts = [(p[0] * target_width, p[1] * target_height) for p in ref_pts]
    crop_size = (target_width, target_height)

    s = np.array(ref_pts).astype(np.float32)
    r = np.array(landmarks).astype(np.float32)

    tfm = SimilarityTransform()
    tfm.estimate(r, s)
    M = tfm.params[0:2, :]
    face_img = cv2.warpAffine(src_img, M, crop_size)

    return face_img


def extract_face(img_np, landmarks, target_width, target_height):
    if len(landmarks) != 1:
        img_pil = Image.fromarray(img_np).resize((target_width, target_height), Image.ANTIALIAS)
        return np.array(img_pil)

    indices = [LEFT_EYE_IDX, RIGHT_EYE_IDX, NOSE_IDX]
    landmarks = landmarks[0][indices]

    return alignment(img_np, landmarks, target_width, target_height)


def extract_landmarks(img_np, fa=None):
    if fa is None:
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, face_detector='blazeface')
    if img_np.ndim == 3:
        img_np = np.expand_dims(img_np, axis=0)
    img_np = img_np.transpose(0, 3, 1, 2)
    img_tensor = torch.Tensor(img_np)
    return fa.get_landmarks_from_batch(img_tensor.cuda())
