import cv2
import numpy as np
from tkinter import filedialog

RED_MIN = 0
RED_MAX = 8
RED_MIN2 = 170
RED_MAX2 = 180
RED_SAT_MIN = int(0.5 * 255)
RED_SAT_MAX = 255
RED_VAL_MIN = int(0.6 * 255)
RED_VAL_MAX = 255

BLUE_MIN = 105
BLUE_MAX = 125
BLUE_SAT_MIN = int(0.3 * 255)
BLUE_SAT_MAX = 255
BLUE_VAL_MIN = int(0.7 * 255)
BLUE_VAL_MAX = 255

BOUNDING_MIN_H = 40
BOUNDING_MIN_W = 40

BOUNDING_MAX_H = 80
BOUNDING_MAX_W = 80

index = 0


def opening(img):
    img_copy = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed_img = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, kernel)
    return closed_img

def closing(img):
    img_copy = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    closed_img = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)
    return closed_img


def threshold_image(img, min_val, max_val):

    # threshold the HSV image
    mask = cv2.inRange(img, min_val, max_val)
    return mask


def fill_image(img):
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = img.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    img_floodfill = img.copy()
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)

    # invert floodfilled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)

    # combine the two images to get the foreground
    floodfill_result = (img | img_floodfill_inv)

    return floodfill_result

def extract_sign(original_img, img):
    global index

    # Find contours
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    # cnt = contours[0]
    padding = 2

    foundSign = False
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(original_img, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)
        # print("H, w: %f %f" % (h, w))
        if (h >= BOUNDING_MIN_H and h <= BOUNDING_MAX_H and
                w >= BOUNDING_MIN_W and w <= BOUNDING_MAX_W):

            roi = original_img[y:y + h + padding, x:x + w + padding]
            path = "sign" + str(index) + ".png"
            index += 1
            # cv2.imshow("ROI", roi)
            cv2.imwrite(path, roi)
            foundSign = True

    cv2.imshow("image", original_img)
    path = "boundingboxes" + str(index) + ".png"
    cv2.imwrite(path, original_img)
    return foundSign



def detect_red_sign(img):
    # define range of red in HSV
    lower_red = np.array([RED_MIN, RED_SAT_MIN, RED_VAL_MIN])
    upper_red = np.array([RED_MAX, RED_SAT_MAX, RED_VAL_MAX])
    lower_red2 = np.array([RED_MIN2, RED_SAT_MIN, RED_VAL_MIN])
    upper_red2 = np.array([RED_MAX2, RED_SAT_MAX, RED_VAL_MAX])

    red_mask = threshold_image(img, lower_red, upper_red)
    red_mask2 = threshold_image(img, lower_red2, upper_red2)
    cv2.bitwise_or(red_mask, red_mask2, red_mask)

    # Remove noise by applying opening operation

    cv2.imshow("Thresholding", red_mask)

    mask_no_noise = closing(red_mask)
    # mask_no_noise = opening(red_mask)

    cv2.imshow("red closing", mask_no_noise)
    floodfilled_img = fill_image(mask_no_noise)
    # cv2.imshow("floodfilled", floodfilled_img)
    if extract_sign(img, floodfilled_img):
        return "STOP"
    return "NO_SIGN"



def detect_blue_sign(img):
    # define range of blue in HSV
    lower_blue = np.array([BLUE_MIN, BLUE_SAT_MIN, BLUE_VAL_MIN])
    upper_blue = np.array([BLUE_MAX, BLUE_SAT_MAX, BLUE_VAL_MAX])

    # threshold the HSV image to get only blue colors
    blue_mask = threshold_image(img, lower_blue, upper_blue)

    cv2.imshow("Thresholding blue", blue_mask)

    mask_no_noise = closing(blue_mask)
    # mask_no_noise = opening(blue_mask)

    cv2.imshow("Blue closing", mask_no_noise)
    floodfilled_img = fill_image(mask_no_noise)
    if extract_sign(img, floodfilled_img):
        return "PARKING"
    return "NO_SIGN"


def find_sign(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sign = detect_red_sign(hsv_img)
    if sign == "NO_SIGN":
        sign = detect_blue_sign(hsv_img)
    return sign


def main():
    while 1:
        inpath = filedialog.askopenfilename()
        img = cv2.imread(inpath, cv2.IMREAD_COLOR)

        # resize image
        # img = cv2.resize(img, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("Original image", img)
        h, w = img.shape[:2]
        h = int (0.5 * h)
        x,y = int(0), int(0)
        crop_img = img[x:x+h, y:y+w]
        cv2.imshow("cropped", crop_img)

        sign = find_sign(crop_img)
        print(sign)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()

