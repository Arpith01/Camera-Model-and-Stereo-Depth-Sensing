#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import json
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict


def load_camera_parameters():
    fs_l = cv.FileStorage("../../parameters/left_camera_intrinsics.xml", cv.FILE_STORAGE_READ)
    left_camera_intrinsics = fs_l.getNode("camera_intrinsic").mat()
    left_camera_distortion = fs_l.getNode("camera_distortion").mat()

    fs_r = cv.FileStorage("../../parameters/right_camera_intrinsics.xml", cv.FILE_STORAGE_READ)
    right_camera_intrinsics = fs_r.getNode("camera_intrinsic").mat()
    right_camera_distortion = fs_r.getNode("camera_distortion").mat()

    fs_sc = cv.FileStorage("../../parameters/stereo_calibration.xml", cv.FILE_STORAGE_READ)
    R = fs_sc.getNode("rotation_calib").mat()
    T = fs_sc.getNode("translation_calib").mat()

    fs_sr = cv.FileStorage("../../parameters/stereo_rectification.xml", cv.FILE_STORAGE_READ)
    R1 = fs_sr.getNode("R1_rectify").mat()
    R2 = fs_sr.getNode("R2_rectify").mat()
    P1 = fs_sr.getNode("P1_rectify").mat()
    P2 = fs_sr.getNode("P2_rectify").mat()

    return left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion, R, T, R1, R2, P1, P2

def undistort_images(image_array, intrinsic_matrix, distortion_coefficients, rectification_matrix, new_camera_matrix):
    undistorted_image_array = []
    for img in image_array:
        undistorted_image_array.append(get_undistorted_image(img, intrinsic_matrix, distortion_coefficients, rectification_matrix, new_camera_matrix))
    return undistorted_image_array

def get_undistorted_image(original_image, intrinsic_matrix, distortion_coefficients, rectification_matrix, new_camera_matrix, write_to_file=False):
    h, w = original_image.shape[:2]
    mapx, mapy = cv.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, rectification_matrix, new_camera_matrix, (w,h), 5)
    undistorted_img = cv.remap(original_image, mapx, mapy, cv.INTER_LINEAR)
    if(write_to_file):
        cv.imwrite("undistorted_image.png", undistorted_img)
    return undistorted_img

def plot_images(image1, image2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap = 'gray')
    ax1.set_title('All Key points', fontsize=30)
    ax2.imshow(image2, cmap = 'gray')
    ax2.set_title('Local Maxima points', fontsize=30)
    plt.show()

def join_arrays(keypoints, descriptors):
    keyp_desc = []
    for i in range(len(keypoints)):
        keyp_desc.append((keypoints[i],descriptors[i]))
    return keyp_desc

def split_arrays(keyp_desc):
    keypoints = []
    descriptors = []
    for i in range(len(keyp_desc)):
        keypoints.append(keyp_desc[i][0])
        descriptors.append(keyp_desc[i][1])
    return np.array(keypoints), np.array(descriptors)

def sort_keypoints(keyp_desc):
    return sorted(keyp_desc, key = lambda k: k[0].pt)

def get_localised_max_keypoints(keyp_desc, window_size = 10):
    kp_map = {}
    for kp_s_i in keyp_desc:
        window_x = kp_s_i[0].pt[0]//window_size
        window_y = kp_s_i[0].pt[1]//window_size
        if(window_x in kp_map):
            if(window_y in kp_map[window_x]):
                kp_map[window_x][window_y].append(kp_s_i)
            else:
                kp_map[window_x][window_y] = [kp_s_i]
        else:
            kp_map[window_x] = {}
    kp_local_maximas = []
    for hor_dict in kp_map.values():
        for vert_dict in hor_dict.values():
            kp_local_maximas.append(max(vert_dict, key=lambda kp: kp[0].response))
    return kp_local_maximas



def scatter_plot(matches,kp_l, kp_r, P1, P2):
    points1 = np.zeros([len(matches),1,2])
    points2 = np.zeros([len(matches),1,2])
    for i in range(len(matches)):
        points1[i][0] = np.array(kp_l[matches[i].queryIdx].pt)
        points2[i][0] = np.array(kp_r[matches[i].trainIdx].pt)
    
    points1 = np.array(points1)
    points2 = np.array(points2)
    points1_re = np.row_stack((points1[:,0,0], points1[:,0,1]))
    points2_re = np.row_stack((points2[:,0,0], points2[:,0,1]))

    triangulate = cv.triangulatePoints(P1,P2,points1_re,points2_re)
    x,y,z,w = triangulate

    fig = plt.figure(figsize=(24, 9))
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(x/w,y/w,z/w)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def refine_matches(matches, kp_l, kp_r):
    cleared_matches = []
    for i in range(len(matches)):
        point1 = kp_l[matches[i].queryIdx].pt
        point2 = kp_r[matches[i].trainIdx].pt
        if(abs(point1[1] - point2[1]))<10:
            cleared_matches.append(matches[i])
    cleared_matches = sorted(cleared_matches, key = lambda x:x.distance)
    return cleared_matches


if __name__ == "__main__":
    left_imgs = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob("../../images/task_3_and_4/left_1.png")]
    right_imgs = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob("../../images/task_3_and_4/right_1.png")] 
    left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion, R, T, R1, R2, P1, P2 = load_camera_parameters()
    left_imgs_undistorted = undistort_images(left_imgs, left_camera_intrinsics, left_camera_distortion, R1, P1)
    right_imgs_undistorted = undistort_images(right_imgs, right_camera_intrinsics, right_camera_distortion, R2, P2)
    
    l_img = left_imgs_undistorted[0]
    r_img = right_imgs_undistorted[0]

    orb = cv.ORB_create()
    kp_l, des_l = orb.detectAndCompute(l_img,None)
    kp_r, des_r = orb.detectAndCompute(r_img,None)
    dummy_l = left_imgs_undistorted[0]
    dummy_r = right_imgs_undistorted[0]

    kp_desc_l = join_arrays(kp_l, des_l)
    kp_desc_r = join_arrays(kp_r, des_r)
    kp_desc_l_max = get_localised_max_keypoints(kp_desc_l)
    kp_desc_r_max = get_localised_max_keypoints(kp_desc_r)
    kp_l_max, desc_l_max = split_arrays(kp_desc_l_max)
    kp_r_max, desc_r_max = split_arrays(kp_desc_r_max)

    l_img_kp = cv.drawKeypoints(l_img, kp_l_max, l_img.copy(), color=(0,255,0), flags=0)
    r_img_kp = cv.drawKeypoints(r_img, kp_r_max, r_img.copy(), color=(0,255,0), flags=0)
    plot_images(l_img_kp,r_img_kp)
    
    bf = cv.BFMatcher_create(normType = cv.NORM_HAMMING)
    matches = bf.match(desc_l_max,desc_r_max)
    matches = refine_matches(matches, kp_l_max, kp_r_max)
    img3 = None
    img3 = cv.drawMatches(l_img,kp_l_max,r_img,kp_r_max,matches, img3, flags=2)
    plt.figure(figsize=(24, 9))
    plt.imshow(img3, cmap = 'gray')
    plt.show()

    scatter_plot(matches,kp_l_max, kp_r_max, P1, P2)
