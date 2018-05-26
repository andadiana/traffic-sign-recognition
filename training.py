import os
import cv2
import numpy as np
import math
import operator
from scipy.spatial import distance
from matplotlib import pyplot as plt
import pyximport; pyximport.install()
import hog


TRAIN_DATA_DIR = "Training/"
TEST_DATA_DIR = "Testing/"

GOOD_SIGNS = ['00013', '00019', '00021', '00022', '00028', '00034', '00036', '00037', '00040', '00041', '00045',
              '00053', '00054', '00056', '00061']

classes = {}


def build_classes_images(images, labels):
    unique_labels = set(labels)
    for label in unique_labels:
        # Pick the first image for each label
        image = images[labels.index(label)]
        classes[label] = image


# Display the first image of each label
def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(15, 15))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label
        image = images[labels.index(label)]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.subplot(8, 8, i)  # A grid of 8 rows x 8 columns
        plt.axis('off')
        plt.title("Label {0} ({1})".format(label, labels.count(label)))
        i += 1
        _ = plt.imshow(image)
    plt.show()


def load_data(data_dir):
    # Get all subdirectories of data_dir. Each represents a label.
    directories = [d for d in os.listdir(data_dir)
                   if os.path.isdir(os.path.join(data_dir, d))]
    # Loop through the label directories and collect the data in
    # two lists, labels and images.
    labels = []
    images = []
    dim = (32, 32)
    for d in directories:
        if d in GOOD_SIGNS:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f)
                          for f in os.listdir(label_dir)
                          if f.endswith(".ppm")]
            for f in file_names:
                img = cv2.imread(f)
                resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
                images.append(resized_img)
                labels.append(int(d))
    return images, labels


def get_hog_bin(hog_intervals, angle):
    # print(angle)
    for i in range(len(hog_intervals)):
        interval = hog_intervals[i]
        if angle < interval:
            return i
    return len(hog_intervals)


# def hog_v(img, starti, startj, h, w):
#     imgh, imgw = np.shape(img)
#     hog_vector = [0] * 9
#     intervals_hog = [x for x in range(20, 180, 20)]
#
#     for i in range(starti, starti + h):
#         for j in range(startj, startj + w):
#             gradx = int(img[i][j])
#             grady = int(img[i][j])
#             if i != 0 and i != imgh - 1:
#                 grady = int(img[i + 1][j]) - int(img[i - 1][j])
#             if j != 0 and j != imgw - 1:
#                 gradx = int(img[i][j + 1]) - int(img[i][j - 1])
#             magnitude = math.sqrt(gradx * gradx + grady * grady)
#             angle = math.atan2(grady, gradx)
#             if angle < 0:
#                 angle += math.pi
#             angle_deg = (angle * 180) / math.pi
#             hog_bin = get_hog_bin(intervals_hog, angle_deg)
#             hog_vector[hog_bin] += magnitude
#
#     return hog_vector


def hog_feature_vec(img):
    h, w = np.shape(img)

    m = h // 4
    n = w // 4

    feature_vec = []
    for i in range(0, h, m):
        for j in range(0, w, n):
            # hog_vector = hog_vec(img, i, j, m, n)
            hog_vec = hog.hog(img, i, j, m, n)
            feature_vec += hog_vec
    return np.array(feature_vec)


def get_neighbors(test_feature_vec, train_feature_vecs, train_labels, k):
    distances = []
    for i in range(len(train_feature_vecs)):
        dist = distance.euclidean(test_feature_vec, train_feature_vecs[i])
        distances.append((train_labels[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])
    return neighbors


def predict_label(test_feature_vec, train_feature_vecs, train_labels):
    min_dist = distance.euclidean(test_feature_vec, train_feature_vecs[0])
    min_index = 0
    for i in range(1, len(train_feature_vecs)):
        dist = distance.euclidean(test_feature_vec, train_feature_vecs[i])
        if dist < min_dist:
            min_dist = dist
            min_index = i
    return train_labels[min_index]


def predict_label_knn(test_feature_vec, train_feature_vecs, train_labels, k):
    class_votes = {}
    neighbors = get_neighbors(test_feature_vec, train_feature_vecs, train_labels, k)

    for i in range(len(neighbors)):
        label = neighbors[i]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def convert_grayscale(images):
    gray_images = []
    for img in images:
        gray_images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return gray_images


def knn(train_images, train_labels, test_images, test_labels):
    train_feature_vecs = []
    for train_img in train_images:
        feature_vector = hog_feature_vec(train_img)
        train_feature_vecs.append(feature_vector)

    predicted_labels_test = []
    i = 0
    for test_img in test_images:
        feature_vector = hog_feature_vec(test_img)
        predicted_label = predict_label_knn(feature_vector, train_feature_vecs, train_labels, k=3)
        print("Predicted: ", predicted_label, " index: ", i)
        i += 1
        predicted_labels_test.append(predicted_label)

    nr_correct = 0
    for i in range(len(predicted_labels_test)):
        print("Test label: ", test_labels[i], "Predicted label: ", predicted_labels_test[i])
        if test_labels[i] == predicted_labels_test[i]:
            nr_correct += 1

    accuracy = (nr_correct / len(test_images)) * 100
    print("Accuracy is: ", accuracy)


def train():
    train_images_original, train_labels = load_data(TRAIN_DATA_DIR)
    build_classes_images(train_images_original, train_labels)
    train_images = convert_grayscale(train_images_original)

    train_feature_vecs = []
    for train_img in train_images:
        feature_vector = hog_feature_vec(train_img)
        train_feature_vecs.append(feature_vector)

    return train_feature_vecs, train_labels


def predict(img, train_feature_vecs, train_labels):
    dim = (32, 32)
    resized_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    grayscale_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    feature_vector = hog_feature_vec(grayscale_img)
    predicted_label = predict_label_knn(feature_vector, train_feature_vecs, train_labels, k=3)
    return classes[predicted_label], predicted_label


def main():

    train_images_original, train_labels = load_data(TRAIN_DATA_DIR)
    test_images_original, test_labels = load_data(TEST_DATA_DIR)
    display_images_and_labels(train_images_original, train_labels)

    train_images = convert_grayscale(train_images_original)
    test_images = convert_grayscale(test_images_original)

    nr_images_train = len(train_images)
    print("Number of training images: ", nr_images_train)
    nr_images_test = len(test_images)
    print("Number of test images: ", nr_images_test)

    knn(train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    main()
