import cv2
import math
import numpy as np

from matplotlib import pyplot as plt

def Histogram_Matching(image,target):

    # 이미지 크기 받기
    H, W = image.shape
    P, Q = target.shape

    # plt 축 설정
    f, axes = plt.subplots(4,1)

    # histogram 값 얻기
    n_r,r, p_r = axes[0].hist(image.ravel(), 256, [0, 256])
    n_z,z ,p_z = axes[1].hist(target.ravel(), 256,[0,256])

    # pdf 초기화
    pdf_r = [x for x in range(256)]
    pdf_z = [x for x in range(256)]

    # pdf 값 입력
    for i in range(256):
        pdf_r[i] = n_r[i] / (H*W)
        pdf_z[i] = n_z[i] / (P*Q)

    # cdf 초기화
    cdf_r = np.zeros(256)
    cdf_z = np.zeros(256)

    # cdf 값 입력
    for i in range(256):
        sum_r = 0.0
        sum_z = 0.0
        for j in range(i):
            sum_r += pdf_r[j]
            sum_z += pdf_z[j]
        cdf_r[i] = np.round(255 * sum_r)
        cdf_z[i] = np.round(255 * sum_z)

    # 역함수 초기화
    Inverse_G = [x for x in range(256)]

    # 역함수 값 입력
    for i in range(256):
        Inverse_G[int(cdf_z[i])] = i

    # 출력 이미지 초기화
    output = np.zeros((H,W),dtype = np.uint8)

    # 출력 이미지 생성
    for i in range(H):
        for j in range(W):
            output[i,j] = Inverse_G[int(cdf_r[int(image[i,j])])]

    return output