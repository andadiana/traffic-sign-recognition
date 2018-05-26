import numpy as np
import math


cpdef get_hog_bin(hog_intervals, angle):
    for i in range(len(hog_intervals)):
        interval = hog_intervals[i]
        if angle < interval:
            return i
    return len(hog_intervals)


cpdef hog(unsigned char [:, :] img, int starti, int startj, int h, int w):
    cdef int imgh, imgw, gradx, grady, hog_bin
    cdef float magnitude, angle, angle_deg
    imgh, imgw = np.shape(img)
    hog_vector = [0] * 9
    intervals_hog = [x for x in range(20, 180, 20)]

    for i in range(starti, starti + h):
        for j in range(startj, startj + w):
            gradx = int(img[i][j])
            grady = int(img[i][j])
            if i != 0 and i != imgh - 1:
                grady = int(img[i + 1][j]) - int(img[i - 1][j])
            if j != 0 and j != imgw - 1:
                gradx = int(img[i][j + 1]) - int(img[i][j - 1])
            magnitude = math.sqrt(gradx * gradx + grady * grady)
            angle = math.atan2(grady, gradx)
            if angle < 0:
                angle += math.pi
            angle_deg = (angle * 180) / math.pi
            hog_bin = get_hog_bin(intervals_hog, angle_deg)
            hog_vector[hog_bin] += magnitude

    return hog_vector