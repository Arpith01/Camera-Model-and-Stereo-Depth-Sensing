#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import json
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    images_left = glob.glob("../../images/task_3_and_4/left*.png")
    images_right = glob.glob("../../images/task_3_and_4/right*.png")
    images_left.sort()
    images_right.sort()

    fs_l = cv2.FileStorage("../../parameters/left_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
    left_camera_intrinsics = fs_l.getNode("camera_intrinsic").mat()
    left_camera_distortion = fs_l.getNode("camera_distortion").mat()

    fs_r = cv2.FileStorage("../../parameters/right_camera_intrinsics.xml", cv2.FILE_STORAGE_READ)
    right_camera_intrinsics = fs_r.getNode("camera_intrinsic").mat()
    right_camera_distortion = fs_r.getNode("camera_distortion").mat()

    fs_sc = cv2.FileStorage("../../parameters/stereo_calibration.xml", cv2.FILE_STORAGE_READ)
    R = fs_sc.getNode("rotation_calib").mat()
    T = fs_sc.getNode("translation_calib").mat()

    fs_sr = cv2.FileStorage("../../parameters/stereo_rectification.xml", cv2.FILE_STORAGE_READ)
    R1 = fs_sr.getNode("R1_rectify").mat()
    R2 = fs_sr.getNode("R2_rectify").mat()
    P1 = fs_sr.getNode("P1_rectify").mat()
    P2 = fs_sr.getNode("P2_rectify").mat()
    Q = fs_sr.getNode("Q_rectify").mat()


    left_maps = cv2.initUndistortRectifyMap(left_camera_intrinsics, left_camera_distortion, R1, P1, (640,480), 5)
    right_maps = cv2.initUndistortRectifyMap(right_camera_intrinsics, right_camera_distortion, R2, P2, (640, 480), 5)

    for i in range(len(images_left)):
        l_img = cv2.imread(images_left[i])
        r_img = cv2.imread(images_right[i])
        left_img_remap = cv2.remap(l_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
        right_img_remap = cv2.remap(r_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
        gray1 = cv2.cvtColor(left_img_remap, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(right_img_remap, cv2.COLOR_BGR2GRAY)
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=13)
        disparity = stereo.compute(gray1,gray2)
        depth = cv2.reprojectImageTo3D(disparity, Q)
        plt.imshow(disparity, 'gray')
        plt.show()
        if i == 3 or i == 1:
            cv2.imwrite("../../output/task_4/image" +str(i)+".png", disparity)
