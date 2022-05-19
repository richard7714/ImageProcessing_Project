import numpy as np
import cv2
from matplotlib import pyplot as plt

def gamma(image,gamma):
    # 이미지 읽기
    H,W = image.shape

    # 이미지 정규화
    normalized_img = image/255

    # 감마 보정 파라미터 설정
    c_param = gamma

    corrected_image = np.zeros((H,W))

    # 감마 보정
    for i in range(H):
        for j in range(W):
            corrected_image[i,j] = normalized_img[i,j] ** c_param

    # 화소 맵핑
    corrected_image = (corrected_image * 255).astype(np.uint8)

    return corrected_image