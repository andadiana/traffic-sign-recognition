import cv2
import numpy as np
from matplotlib import pyplot as plt
from tkinter import filedialog
import training

filename = "Images/prioritate.jpg"


RED_MIN = 0
RED_MAX = 8
RED_MIN2 = 170
RED_MAX2 = 180
RED_SAT_MIN = int(0.5 * 255)
RED_SAT_MAX = 255
RED_VAL_MIN = int(0.5 * 255)
RED_VAL_MAX = 255

BLUE_MIN = 100
BLUE_MAX = 120
BLUE_SAT_MIN = int(0.6 * 255)
BLUE_SAT_MAX = 255
BLUE_VAL_MIN = int(0.5 * 255)
BLUE_VAL_MAX = 255

YELLOW_MIN = 20
YELLOW_MAX = 33
YELLOW_SAT_MIN = int(0.4 * 255)
YELLOW_SAT_MAX = 255
YELLOW_VAL_MIN = int(0.7 * 255)
YELLOW_VAL_MAX = 255

BOUNDING_MIN_H = 40
BOUNDING_MIN_W = 40

BOUNDING_MAX_H = 250
BOUNDING_MAX_W = 250

index = 0

ideal_mean = 125


def histogram_mean(hist):
    mean = 0
    s = 0
    for (x,_), value in np.ndenumerate(hist):
        # print(x, y, value)
        mean += x * value
        s += value
    return mean / s


def compute_histogram_hsv(hsv_img):
    hist_v = cv2.calcHist([hsv_img], [2], None, [256], [0, 256])
    # Plot histogram
    plt.plot(hist_v, color='b')
    plt.xlim([0, 256])
    plt.show()

    return hist_v


def slide_brightness(hsv_img, offset):
    h, s, v = cv2.split(hsv_img)
    print("Value array")
    print(v.dtype)

    # v += offset
    # v_new = np.clip(v, 0, 255)
    # v_new = np.uint8(v_new)
    v_new = np.where((v + offset) > 255, 255, v + offset)
    v_new = v_new.astype(np.uint8)
    return cv2.merge((h, s, v_new))


def adjust_brightness(hsv_img):
    hist_v = compute_histogram_hsv(hsv_img)
    mean = histogram_mean(hist_v)
    print("Mean is ", mean)

    new_img = hsv_img
    if mean < 100:
        new_img = slide_brightness(hsv_img, ideal_mean - mean)
    adjusted = cv2.cvtColor(new_img, cv2.COLOR_HSV2BGR)
    # TODO not working
    cv2.imshow("Adjusted brightness", adjusted)
    return new_img


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


def extract_sign(original_img, img, padding):
    global index

    # Find contours
    im2, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), 1)

    # cnt = contours[0]
    # padding = 4

    found_sign = False
    sign = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # print("H, w: %f %f" % (h, w))
        if (h >= BOUNDING_MIN_H and h <= BOUNDING_MAX_H and
                w >= BOUNDING_MIN_W and w <= BOUNDING_MAX_W):
            cv2.rectangle(original_img, (x - padding, y - padding), (x + w + padding, y + h + padding), (0, 255, 0), 2)
            print("ROI h, w: ", h, w)
            roi = original_img[y - padding:y + h + padding, x - padding:x + w + padding]
            index += 1
            # cv2.imshow("ROI", roi)
            sign = roi
            found_sign = True

    cv2.imshow("image", original_img)
    return sign, found_sign


def detect_red_sign(original_img, img):
    # define range of red in HSV
    lower_red = np.array([RED_MIN, RED_SAT_MIN, RED_VAL_MIN])
    upper_red = np.array([RED_MAX, RED_SAT_MAX, RED_VAL_MAX])
    lower_red2 = np.array([RED_MIN2, RED_SAT_MIN, RED_VAL_MIN])
    upper_red2 = np.array([RED_MAX2, RED_SAT_MAX, RED_VAL_MAX])

    red_mask = threshold_image(img, lower_red, upper_red)
    red_mask2 = threshold_image(img, lower_red2, upper_red2)
    cv2.bitwise_or(red_mask, red_mask2, red_mask)

    cv2.imshow("Thresholding", red_mask)

    # Remove noise by applying closing operation
    mask_no_noise = closing(red_mask)
    # cv2.imshow("red closing", mask_no_noise)

    floodfilled_img = fill_image(mask_no_noise)
    cv2.imshow("floodfilled", floodfilled_img)
    sign, found_sign = extract_sign(original_img, floodfilled_img, 2)
    if found_sign:
        return sign, "STOP"
    return None, "NO_SIGN"


def detect_blue_sign(original_img, img):
    # define range of blue in HSV
    lower_blue = np.array([BLUE_MIN, BLUE_SAT_MIN, BLUE_VAL_MIN])
    upper_blue = np.array([BLUE_MAX, BLUE_SAT_MAX, BLUE_VAL_MAX])

    # threshold the HSV image to get only blue colors
    blue_mask = threshold_image(img, lower_blue, upper_blue)

    cv2.imshow("Thresholding blue", blue_mask)

    # Remove noise by applying closing operation
    mask_no_noise = closing(blue_mask)
    # cv2.imshow("Blue closing", mask_no_noise)

    floodfilled_img = fill_image(mask_no_noise)
    cv2.imshow("floodfilled", floodfilled_img)
    sign, found_sign = extract_sign(original_img, floodfilled_img, 2)
    if found_sign:
        return sign, "PARKING"
    return None, "NO_SIGN"


def detect_yellow_sign(original_img, img):
    # define range of red in HSV
    lower_yellow = np.array([YELLOW_MIN, YELLOW_SAT_MIN, YELLOW_VAL_MIN])
    upper_yellow = np.array([YELLOW_MAX, YELLOW_SAT_MAX, YELLOW_VAL_MAX])

    yellow_mask = threshold_image(img, lower_yellow, upper_yellow)

    cv2.imshow("Thresholding", yellow_mask)

    # Remove noise by applying closing operation
    mask_no_noise = closing(yellow_mask)\

    floodfilled_img = fill_image(mask_no_noise)
    cv2.imshow("floodfilled", floodfilled_img)
    sign, found_sign = extract_sign(original_img, floodfilled_img, 15)
    if found_sign:
        return sign, "YELLOW"
    return None, "NO_SIGN"


def find_sign(img):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # hsv_img = adjust_brightness(hsv_img)

    sign, sign_name = detect_red_sign(img, hsv_img)
    if sign_name == "NO_SIGN":
        sign, sign_name = detect_blue_sign(img, hsv_img)
        if sign_name == "NO_SIGN":
            sign, sign_name = detect_yellow_sign(img, hsv_img)
    return sign, sign_name


def main():
    train_feature_vecs, train_labels = training.train()
    while 1:
        inpath = filedialog.askopenfilename()
        img = cv2.imread(inpath, cv2.IMREAD_COLOR)

        # img = cv2.imread(filename, cv2.IMREAD_COLOR)
        sign, sign_name = find_sign(img)
        if sign is not None:
            cv2.imshow("Sign", sign)
            class_image, label = training.predict(sign, train_feature_vecs, train_labels)
            print("Predicted label is: ", label)
            cv2.imshow("Class", class_image)
        print(sign_name)

        cv2.waitKey(0)


if __name__ == "__main__":
    main()

