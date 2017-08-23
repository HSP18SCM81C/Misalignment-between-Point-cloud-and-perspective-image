import numpy as np
import cv2
import math


def drawMatches(img1, kp1, img2, kp2, matches):


    if len(img1.shape) == 3:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1], img1.shape[2])
    elif len(img1.shape) == 2:
        new_shape = (max(img1.shape[0], img2.shape[0]), img1.shape[1] + img2.shape[1])
    new_img = np.zeros(new_shape, type(img1.flat[0]))
    # Place images onto the new image.
    new_img[0:img1.shape[0], 0:img1.shape[1]] = img1
    new_img[0:img2.shape[0], img1.shape[1]:img1.shape[1] + img2.shape[1]] = img2
    angles = []
    for m in matches:

        end1 = tuple(np.round(kp1[m.trainIdx].pt).astype(int))
        end2 = tuple(np.round(kp2[m.queryIdx].pt).astype(int) + np.array([img1.shape[1], 0]))
        angles.append(math.atan((float)(end2[1] - end1[1]) / (end1[0] - end2[0])) * (180 / math.pi))

    print(sum(angles) / len(angles))



img2 = cv2.imread("front.jpg");
img1 = cv2.imread("front.png");

orb = cv2.ORB_create(1000, 1.2)

# Detect keypoints of original image
(kp1, des1) = orb.detectAndCompute(img1, None)

# Detect keypoints of rotated image
(kp2, des2) = orb.detectAndCompute(img2, None)

# Create matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Do matching
matches = bf.match(des1, des2)

# Sort the matches based on distance.  Least distance
# is better
matches = sorted(matches, key=lambda val: val.distance)

# Show only the top 10 matches
#drawMatches(img1, kp1, img2, kp2, matches)


