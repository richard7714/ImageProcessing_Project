import cv2
import numpy as np


def SIFT(img, original):
    gray1 = img
    gray2 = original

    # SIFT 추출기 생성
    sift = cv2.ORB_create()
    # 키 포인트 검출과 서술자 계산
    kp1, desc1 = sift.detectAndCompute(gray1, None)
    kp2, desc2 = sift.detectAndCompute(gray2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)
    matches = matcher.match(desc1, desc2)

    matches = sorted(matches, key=lambda x: x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])

    mtrx, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    h, w = gray1.shape[:2]
    pts = np.float32([ [[0, 0]], [[0, h - 1]], [[w - 1, h - 1]], [[w - 1, 0]]] )
    dst = cv2.perspectiveTransform(pts, mtrx)

    matchesMask = mask.ravel().tolist()
    res = cv2.drawMatches(gray1, kp1, gray2, kp2, matches, None, \
                          matchesMask=matchesMask)

    accuracy = round(float(mask.sum()) / mask.size,1)
    # 키 포인트 그리기
    img_draw = cv2.putText(res, str(mask.sum())+' '+str(accuracy), (0, img.shape[0]), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.5,
                           color=(250, 255, 100))

    return img_draw
