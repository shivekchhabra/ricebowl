import cv2
import os
import numpy as np
from imutils import paths


# Overview:
# This file contains the code for generalised processing of images

# Reading image from path
def read_image(path):
    img = cv2.imread(path)
    return img


# Displaying an image
def show_image(img, title='img'):
    cv2.imshow(title, img)
    cv2.waitKey(0)  # Press any key to quit.


# Writing an image
def write_image(filepath, img):
    cv2.imwrite(filepath, img)


# Inverting (converting to negative)
def inverting(img):
    invert = 255 - img
    return invert


# Making an image grayscale
def gray_scale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# Resizing image according to the length and width specified.
def resize(img, length, width):
    img = cv2.resize(img, (length, width))
    return img


# Applying Gaussian blurring according to the kernel specified. (Smoothing of image)
def gaussian_blurring(img, ksize=(21, 21)):
    blur = cv2.GaussianBlur(img, ksize=ksize, sigmaX=0, sigmaY=0)
    return blur


# Extracting features using ORB (Oriented Fast and Rotated Brief)
def orb_features(img):
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    feat = cv2.drawKeypoints(img, kp, None)
    return feat, des


# Returns the array data of images and their labels (entire path)
def get_images(path):
    path = list(paths.list_images(path))
    data = []
    labels = []
    for imagePath in path:
        label = imagePath.split(os.path.sep)[-2]
        image = read_image(imagePath)
        data.append(image)
        labels.append(label)
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels
