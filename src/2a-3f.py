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
    left_camera_intrinsics = np.loadtxt("Parameters/left_camera_intrinsic.csv", delimiter=",")
    left_camera_distortion = np.loadtxt("Parameters/left_camera_distortion.csv", delimiter=",")
    right_camera_intrinsics = np.loadtxt("Parameters/right_camera_intrinsic.csv", delimiter=",")
    right_camera_distortion = np.loadtxt("Parameters/right_camera_distortion.csv", delimiter=",")
    R = np.loadtxt("Parameters/rotation_calib.csv", delimiter=",")
    T = np.loadtxt("Parameters/translation_calib.csv", delimiter=",")
    R1 = np.loadtxt("Parameters/R1_rectify.csv", delimiter=",")
    R2 = np.loadtxt("Parameters/R2_rectify.csv", delimiter=",")
    P1 = np.loadtxt("Parameters/P1_rectify.csv", delimiter=",")
    P2 = np.loadtxt("Parameters/P2_rectify.csv", delimiter=",")

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

def plot_images(original_image, undistorted_image):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(original_image, cmap = 'gray')
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(undistorted_image, cmap = 'gray')
    ax2.set_title('Undistorted Image', fontsize=50)
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


def get_localised_max_keypoints(keyp_desc):
    window_size = 10
    kp_s = sort_keypoints(keyp_desc)
    map = defaultdict(dict)
    window_x = 0
    for x in range(0,640,window_size):
        window_start = x
        window_end = x+ window_size
        map[window_x] = defaultdict(list)
        for i in range(len(kp_s)):
            kp = kp_s[i][0]
            x = kp_s[i][0].pt[0]
            if(x> window_start and x < window_end):
                # print("window_x = ", window_x , "x = ", x)
                y = kp_s[i][0].pt[1]
                window_y = y//window_size
                map[window_x][window_y].append(kp_s[i])
                # print("window_y = ", window_y)
        window_x+=1
    kp_modified = []
    for x_dict in map.values():
        for y_dict in x_dict.values():
            # print(max(vert_dict, key=lambda kp: kp.response).pt)
            kp_modified.append(max(y_dict, key=lambda kp: kp[0].response))
    return kp_modified

def get_localised_max_keypoints_2(keyp_desc, window_size = 10):
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
    # print(kp_map)
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

    triangulate = cv.triangulatePoints(P1,P2,points1,points2)
    x,y,z,w = triangulate
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(*triangulate)
    plt.show()

def refine_matches(matches, kp_l, kp_r):
    cleared_matches = []
    for i in range(len(matches)):
        point1 = kp_l[matches[i].queryIdx].pt
        point2 = kp_r[matches[i].trainIdx].pt
        if(abs(point1[1] - point2[1]))<15:
            cleared_matches.append(matches[i])
    cleared_matches = sorted(cleared_matches, key = lambda x:x.distance)
    return cleared_matches


if __name__ == "__main__":
    left_imgs = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob("./images/task_3_and_4/left_9.png")]
    right_imgs = [cv.imread(file, cv.IMREAD_GRAYSCALE) for file in glob.glob("./images/task_3_and_4/right_9.png")] 
    left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion, R, T, R1, R2, P1, P2 = load_camera_parameters()
    left_imgs_undistorted = undistort_images(left_imgs, left_camera_intrinsics, left_camera_distortion, R1, P1)
    right_imgs_undistorted = undistort_images(right_imgs, right_camera_intrinsics, right_camera_distortion, R2, P2)
    
    l_img = left_imgs_undistorted[0]
    r_img = right_imgs_undistorted[0]

    orb = cv.ORB_create()
    kp_l, des_l = orb.detectAndCompute(l_img,None)
    kp_r, des_r = orb.detectAndCompute(r_img,None)
    # print(des_r[0])
    dummy_l = left_imgs_undistorted[0]
    dummy_r = right_imgs_undistorted[0]
    kp_desc_l = join_arrays(kp_l, des_l)
    kp_desc_r = join_arrays(kp_r, des_r)
    kp_desc_l_max = get_localised_max_keypoints_2(kp_desc_l)
    kp_desc_r_max = get_localised_max_keypoints_2(kp_desc_r)
    kp_l_max, desc_l_max = split_arrays(kp_desc_l_max)
    kp_r_max, desc_r_max = split_arrays(kp_desc_r_max)

    l_img_kp = cv.drawKeypoints(l_img, kp_l_max, dummy_l, color=(0,255,0), flags=0)
    r_img_kp = cv.drawKeypoints(r_img, kp_r_max, dummy_r, color=(0,255,0), flags=0)
    plot_images(l_img_kp,r_img_kp)
    
    bf = cv.BFMatcher_create(normType = cv.NORM_HAMMING)
    matches = bf.match(desc_l_max,desc_r_max)
    matches = refine_matches(matches, kp_l_max, kp_r_max)
    img3 = None
    img3 = cv.drawMatches(l_img,kp_l_max,r_img,kp_r_max,matches, img3, flags=2)
    plt.imshow(img3, cmap = 'gray')
    plt.show()
    scatter_plot(matches,kp_l_max, kp_r_max, P1, P2)
    print(P1, P2)