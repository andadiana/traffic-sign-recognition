import cv2
import numpy as np

# STOP signs
FILENAME = "Images/30850175_1682513215165350_1569870497_o.jpg"
# FILENAME = "Images/30946199_1682513128498692_1829619125_o.jpg"  # tricou maha
# FILENAME = "Images/30952612_1682513285165343_1770592685_o.jpg"  # floare blugi

RED_MIN = 0
RED_MAX = 10
RED_MIN2 = 170
RED_MAX2 = 180
RED_SAT_MIN = 70
RED_SAT_MAX = 255
RED_VAL_MIN = 70
RED_VAL_MAX = 255

BLUE_MIN = 105
BLUE_MAX = 125
BLUE_SAT_MIN = 25
BLUE_SAT_MAX = 255
BLUE_VAL_MIN = 30
BLUE_VAL_MAX = 255

YELLOW_MIN = 20
YELLOW_MAX = 30
YELLOW_SAT_MIN = 25
YELLOW_SAT_MAX = 255
YELLOW_VAL_MIN = 30
YELLOW_VAL_MAX = 255


def closing(img):
    img_copy = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed_img = cv2.morphologyEx(img_copy, cv2.MORPH_CLOSE, kernel)
    return closed_img


def opening(img):
    img_copy = img.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 1))
    closed_img = cv2.morphologyEx(img_copy, cv2.MORPH_OPEN, kernel)
    return closed_img


def color_segmentation(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of red in HSV
    lower_red = np.array([RED_MIN, RED_SAT_MIN, RED_VAL_MIN])
    upper_red = np.array([RED_MAX, RED_SAT_MAX, RED_VAL_MAX])
    lower_red2 = np.array([RED_MIN2, RED_SAT_MIN, RED_VAL_MIN])
    upper_red2 = np.array([RED_MAX2, RED_SAT_MAX, RED_VAL_MAX])

    # threshold the HSV image to get only red colors
    red_mask = cv2.inRange(hsv_img, lower_red, upper_red)
    red_mask2 = cv2.inRange(hsv_img, lower_red2, upper_red2)
    cv2.bitwise_or(red_mask, red_mask2, red_mask)

    # define range of blue in HSV
    lower_blue = np.array([BLUE_MIN, BLUE_SAT_MIN, BLUE_VAL_MIN])
    upper_blue = np.array([BLUE_MAX, BLUE_SAT_MAX, BLUE_VAL_MAX])

    # threshold the HSV image to get only blue colors
    # blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)

    # TODO: add yellow thresholding

    cv2.imshow("Red mask", red_mask)
    # cv2.imshow("Blue mask", blue_mask)

    # Remove noise by applying opening operation
    red_mask_no_noise = opening(red_mask)
    cv2.imshow("Red mask after opening", red_mask_no_noise)

    # Remove holes in objects by applying closing
    # red_mask_no_noise = closing(red_mask_no_noise)
    # cv2.imshow("Red mask after opening and closing", red_mask_no_noise)

    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = red_mask_no_noise.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    img_floodfill = red_mask_no_noise.copy()
    cv2.floodFill(img_floodfill, mask, (0, 0), 255)

    # invert floodfilled image
    img_floodfill_inv = cv2.bitwise_not(img_floodfill)

    # combine the two images to get the foreground
    floodfill_result = (red_mask_no_noise | img_floodfill_inv)

    cv2.imshow("After flood filling", floodfill_result)

    # Find contours
    im2, contours, hierarchy = cv2.findContours(floodfill_result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

    cv2.imshow("Contours", img)


def main():
    img_original = cv2.imread(FILENAME, cv2.IMREAD_COLOR)

    # resize image
    img = cv2.resize(img_original, (0,0), fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Original image", img)

    color_segmentation(img)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
