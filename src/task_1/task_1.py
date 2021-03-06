#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import json

def get_image_and_object_points(image_array, board_shape=(9,6), scale=25.4):
    image_points = []
    object_points = []
    x,y = board_shape
    obj_points = np.zeros((x*y,3), np.float32)
    obj_points[:, :2] = np.mgrid[0:x,0:y].T.reshape(-1,2)
    obj_points = obj_points * scale
    for img in image_array:
        ret, img_points = cv.findChessboardCorners(img, (x,y), None)
        if(ret):
            image_points.append(img_points)
            object_points.append(obj_points)

    return image_points, object_points


def get_camera_parameters(image_array, board_shape=(9,6), scale=25.4, img_shape=(640, 480)):
    img_points, obj_points = get_image_and_object_points(image_array, board_shape, scale)
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, img_shape, None, None)
    return mtx, dist

def get_undistorted_image(original_image, intrinsic_matrix, distortion_coefficients,write_to_file=False):
    h, w = original_image.shape[:2]
    mapx, mapy = cv.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, None, intrinsic_matrix, (w,h), 5)
    undistorted_img = cv.remap(original_image, mapx, mapy, cv.INTER_LINEAR)
    if(write_to_file):
        cv.imwrite("../../output/undistorted_image.png", undistorted_img)
    return undistorted_img


def plot_images(original_image, chessboard_img, undistorted_image):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_image)
    ax1.set_title('Original Image', fontsize=20)
    ax2.imshow(chessboard_img)
    ax2.set_title('Original Image with corners', fontsize=20)
    ax3.imshow(undistorted_image)
    ax3.set_title('Undistorted Image', fontsize=20)
    plt.show()


def save_camera_parameters(intrinsic_matrix, distortion_coefficients, left= True):
    side = "left" if left else "right"
    fs = cv.FileStorage("../../parameters/" + side + "_camera_intrinsics.xml", cv.FILE_STORAGE_WRITE)
    fs.write('camera_intrinsic', intrinsic_matrix)
    fs.write('camera_distortion', distortion_coefficients)


if __name__ == "__main__":
    left_imgs = [cv.imread(file) for file in sorted(glob.glob("../../images/task_1/left*.png"))]
    right_imgs = [cv.imread(file) for file in sorted(glob.glob("../../images/task_1/right*.png"))]
    left_intrinsic_matrix, left_distortion_coeffecients = get_camera_parameters(left_imgs)
    right_intrinsic_matrix, right_distortion_coefficients = get_camera_parameters(right_imgs)
    save_camera_parameters(left_intrinsic_matrix, left_distortion_coeffecients, True)
    save_camera_parameters(right_intrinsic_matrix, right_distortion_coefficients, False)
    original_image = cv.imread('../../images/task_1/right_2.png')
    undistorted_img = get_undistorted_image(original_image,left_intrinsic_matrix, left_distortion_coeffecients, True)

    _, image_points_1 = cv.findChessboardCorners(original_image, (9,6))
    image_1_corners = cv.drawChessboardCorners(original_image.copy(), (9,6), image_points_1, True)

    plot_images(original_image, image_1_corners, undistorted_img)
