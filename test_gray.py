"""

Image processing programming quiz
(2022-01, ECE4326-001)
---------------------------------
Name: 마승준
Student ID: 12171780
---------------------------------

Problem 3. Implementation the notch filter
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from SIFT import SIFT
from GammaCorrection import gamma


################################################
# You should implement the functions below.    #
# You can create the new function if you need. #
################################################
def get_log_scale(image):
    dft2d_ = np.log(np.abs(image) + 1e-8)
    return dft2d_


def sq_band(image, loc):
    M, N = image.shape
    H = np.ones((M, N))
    U0 = int(M / 2)
    V0 = int(N / 2)

    loc1 = U0 + loc[0]
    loc2 = V0 + loc[1]

    for u in range(M):
        for v in range(N):
            if u >= (loc1 - 10) and u <= (loc1 + 10) and v >= (loc2 - 10) and v <= (loc2 + 10):
                H[u, v] = 0
    return H


def remove_noise(image, cutoff):
    M, N = image.shape
    H, D = np.zeros((M, N)), np.zeros((M, N))

    U0 = int(M / 2)
    V0 = int(N / 2)

    D0 = cutoff

    # For D(u,v)
    for u in range(M):
        for v in range(N):
            u2 = np.power(u, 2)
            v2 = np.power(v, 2)
            D[u, v] = np.sqrt(u2 + v2)

    for u in range(M):
        for v in range(N):
            u_ = np.abs(u - U0)
            v_ = np.abs(v - V0)
            H[u, v] = np.exp(-D[u_, v_] ** 2 / (2 * (D0 ** 2)))

    H = 1 - H

    return H


def pre_before_dft(image):
    H, W = image.shape

    # padding
    padded_image = np.zeros((H * 2, W * 2))
    padded_image[:H, :W] = image

    P, Q = padded_image.shape

    # Centering
    padded_image_new = np.zeros((P, Q))

    for x in range(P):
        for y in range(Q):
            padded_image_new[x, y] = padded_image[x, y] * (-1) ** (x + y)

    return padded_image_new


def post_after_idft(image):
    P, Q = image.shape
    for x in range(P):
        for y in range(Q):
            image[x, y] = image[x, y] * ((-1) ** (x + y))

    output = image[:int(P / 2), :int(Q / 2)].real

    max, min = np.max(output), np.min(output)
    output = ((output - min) / (max - min)) * 255
    return output


#############################################


if __name__ == "__main__":
    '''Code of 3.1. Show the pattern of image in freq. domain.(define pre_before_dft & get_log_scale function)'''
    image = cv2.imread('challenging-60/05_outdoor_hazy.jpg', 0)
    image = cv2.resize(image, (640, 480))

    GT = cv2.imread('challenging-60/05_outdoor_GT.jpg', 0)
    GT = cv2.resize(GT, (640, 480))

    padded_image = pre_before_dft(image)
    dft2d = np.fft.fft2(padded_image)

    # # Get log-scale image to show
    # dft2d_ = get_log_scale(dft2d)

    '''Code of 3.2 - Implementation the denoising filter(define remove_noise function)'''
    H = remove_noise(dft2d, 10)

    '''Code of 3.3 - Return the denoised image(define post_after_idft function)'''
    G = np.multiply(dft2d, H)
    # # Get log-scale image to show
    # dft2d_ = get_log_scale(G)

    idft2d = np.fft.ifft2(G)

    idft2d = post_after_idft(idft2d).astype('uint8')
    #
    # shapen = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])
    #
    # dehazed_img = cv2.filter2D(idft2d,-1,shapen)
    idft2d = gamma(idft2d,0.75)

    image = SIFT(image, GT)
    idft2d = SIFT(idft2d, GT)

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(image, cmap='gray')
    ax1.axis('off')

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(idft2d, cmap='gray'), ax2.axis('off')

    plt.show()

    # dx_before = cv2.bitwise_or(cv2.Sobel(image, -1, 1, 0), cv2.Sobel(image, -1, 0, 1))
    # dx_after = cv2.bitwise_or(cv2.Sobel(idft2d, -1, 1, 0), cv2.Sobel(idft2d, -1, 0, 1))
    #
    # max, min = np.max(dx_before), np.min(dx_before)
    # dx_before = ((dx_before - min) / (max - min)) * 255
    #
    # max, min = np.max(dx_after), np.min(dx_after)
    # dx_after = ((dx_after - min) / (max - min)) * 255
    #
    # fig2 = plt.figure()
    # ax1 = fig2.add_subplot(1, 2, 1)
    # ax1.imshow(dx_before, cmap='gray')
    # ax1.axis('off')
    #
    # ax2 = fig2.add_subplot(1, 2, 2)
    # ax2.imshow(dx_after, cmap='gray'), ax2.axis('off')
    #
    # plt.show()
