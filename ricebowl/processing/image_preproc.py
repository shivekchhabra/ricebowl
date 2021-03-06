import cv2
import os
import math
import pydicom
import numpy as np
from imutils import paths
from scipy import ndimage
from PIL import Image, ImageEnhance, ImageChops


# Overview:
# This file contains the code for generalised processing of images

def read_image(path):
    """
    General function to read an image from a path
    :param path: Path of the image
    :return: Image
    """
    img = cv2.imread(path)
    return img


def show(img):
    """
    General function to display an image
    :param img: Image
    :return: (shows image) (press Q to exit)
    """
    cv2.imshow('', img)
    cv2.waitKey(0)


def write_image(filepath, img):
    """
    General function to save an image
    :param filepath: Path to be saved in
    :param img: Input image
    :return: Image is saved at the desination
    """
    cv2.imwrite(filepath, img)


def inverting(img):
    """
    General function to convert an image to negative
    :param img: Input image
    :return: Inverted image
    """
    invert = 255 - img
    return invert


def gray_scale(img):
    """
    General function to convert an image to grayscale
    :param img: Input image
    :return: Gray scaled image
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    """
    Returns resized image but keeps the aspect ratio intact
    :param image: src image
    :return: resized image
    """
    (h, w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def gaussian_blurring(img, ksize=(21, 21)):
    """
    Applying Gaussian blurring according to the kernel specified. (Smoothing of image)
    :param img: Input image
    :param ksize: Kernel size (as a tuple)
    :return: Blurred image
    """
    blur = cv2.GaussianBlur(img, ksize=ksize, sigmaX=0, sigmaY=0)
    return blur


def orb_features(img):
    """
    Extracting features using ORB (Oriented Fast and Rotated Brief)
    :param img: Input image
    :return: 1. feat - features
             2. des -  descriptor matrix
    """
    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(img, None)
    feat = cv2.drawKeypoints(img, kp, None)
    return feat, des


def get_images(path):
    """
    Finding the array data of images and their labels (entire path)
    :param path: Directory to work on
    :return: 1. data -   array of images
             2. labels - all the image folder names
    """
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


def dcm_to_png(input_directory, output_directory):
    """
    General function to convert .dcm images to .png image format
    :param input_directory: Input directory containing dcm image files
    :param output_directory: Output directory to save all png files
    :return:
    """
    img_list = [f for f in os.listdir(input_directory)]
    total = len(img_list)
    ct = 0
    for i in img_list:
        ct = ct + 1
        print(f'Written image {ct}/{total}')
        if i.endswith('.dcm'):
            ds = pydicom.read_file(input_directory + i)  # reads the image
            img = ds.pixel_array
            try:
                cv2.imwrite(output_directory + i.replace('.dcm', '.png'), img)
            except:
                pass


def denoise(img):
    """
    Removing noise from a colored Image
    :param img: Colored Image
    :return: image
    """
    return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)


def binarization(img):
    """
    Converting an image to binary using Otsu Thresholding
    :param img: Image
    :return: image
    """
    ret, imgf = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY, cv2.THRESH_OTSU)
    return imgf


def erode(img):
    """
    General function to erode text in Image (Performs morphological transformation - erosion)
    :param img: Image
    :return: image
    """
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion


def find_contours(img):
    """
    Finding contours of an image
    :param img: Image
    :return: All contours on an image
    """
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def sharpen(img):
    """
    General function to sharpen an image using a hardcoded kernel
    :param img: Image
    :return: Sharpened image
    """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    return img


def edging(img):
    """
    Used to find edges in gaussian blurred image
    :param img: blurred/smooth image (gaussian blurring)
    :return: image with edges
    """
    canny = cv2.Canny(img, 75, 200)
    return canny


def autorotate(img):
    """
    Autorotate an image according to hough line angle
    :param img: Input image
    :return: Unskewed image at (0deg)
    """
    orig = img.copy()
    img_gray = gray_scale(img)
    img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
    lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180, 100, minLineLength=100, maxLineGap=5)
    angles = []
    for [[x1, y1, x2, y2]] in lines:
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        angles.append(angle)
    median_angle = np.median(angles)
    rotated = ndimage.rotate(orig, median_angle)
    return rotated


def chop_image(image):
    """
    General function to chop image
    :param image: Image
    :return: Image
    """
    bg = Image.new(image.mode, image.size, image.getpixel((0, 0)))
    diff = ImageChops.difference(image, bg)
    bbox = diff.getbbox()
    crop = image.crop(bbox)
    return crop


def image_enhancer(img):
    """
    General function to adjust the image's contrast
    :param img: Image
    :return: Image
    """
    im = Image.fromarray(img)
    enh = ImageEnhance.Contrast(im)
    val = enh.enhance(1.8)
    img = chop_image(val)
    return np.asarray(img)


def remove_shadow(img):
    """
    General function to remove shadows from image
    :param img: Input image
    :return: Output image with no shadows
    """
    rgb_planes = cv2.split(img)
    result_planes = []
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((3, 3), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_planes.append(diff_img)
        result_norm_planes.append(norm_img)

    result = cv2.merge(result_planes)
    result_norm = cv2.merge(result_norm_planes)
    return result_norm
