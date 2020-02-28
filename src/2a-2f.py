#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import glob
import json
from mpl_toolkits.mplot3d import Axes3D



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

def load_camera_parameters():
    left_camera_intrinsics = np.loadtxt("Parameters/left_camera_intrinsic.csv", delimiter=",")
    left_camera_distortion = np.loadtxt("Parameters/left_camera_distortion.csv", delimiter=",")
    right_camera_intrinsics = np.loadtxt("Parameters/right_camera_intrinsic.csv", delimiter=",")
    right_camera_distortion = np.loadtxt("Parameters/right_camera_distortion.csv", delimiter=",")
    return left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion

def stereo_calibrate(obj_points, left_img_points, right_img_points, left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion, img_shape=(640, 480)):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    flags = cv.CALIB_FIX_INTRINSIC
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F  = \
        cv.stereoCalibrate(obj_points,left_img_points,right_img_points, left_camera_intrinsics, left_camera_distortion,\
                 right_camera_intrinsics, right_camera_distortion, img_shape, criteria=criteria,flags=flags)
    camera_model = dict([('M1', cameraMatrix1), ('M2', cameraMatrix2), ('dist1', distCoeffs1),
                            ('dist2', distCoeffs2), ('R', R), ('T', T),
                            ('E', E), ('F', F),('img_shape', img_shape)])
    return camera_model

def stereo_rectify(camera_model, alpha=-1):
    R1, R2, P1, P2, Q, roi1, roi2 = \
        cv.stereoRectify(\
            cameraMatrix1 = camera_model['M1'], distCoeffs1 = camera_model['dist1'], cameraMatrix2 = camera_model['M2'], distCoeffs2 = camera_model['dist2'],\
                 imageSize = camera_model['img_shape'], R = camera_model['R'], T = camera_model['T'], alpha = alpha)
    rectification_params = dict([('R1', R1), ('R2', R2), ('P1', P1),
                            ('P2', P2), ('Q', Q)])
    return rectification_params

def get_undistorted_image(original_image, intrinsic_matrix, distortion_coefficients, rectification_matrix, new_camera_matrix, write_to_file=False):
    h, w = original_image.shape[:2]
    mapx, mapy = cv.initUndistortRectifyMap(intrinsic_matrix, distortion_coefficients, rectification_matrix, new_camera_matrix, (w,h), 5)
    undistorted_img = cv.remap(original_image, mapx, mapy, cv.INTER_LINEAR)
    if(write_to_file):
        cv.imwrite("undistorted_image.png", undistorted_img)
    return undistorted_img

def plot_images(image_1, image_2):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    _, image_points_1 = cv.findChessboardCorners(image_1, (9,6))
    _, image_points_2 = cv.findChessboardCorners(image_2, (9,6))
    # for i, j in zip(image_points_1, image_points_2):
    #     print(i , j)
    image_1_corners = cv.drawChessboardCorners(image_1, (9,6), image_points_1, True)
    image_2_corners = cv.drawChessboardCorners(image_2, (9,6), image_points_2, True)
    ax1.imshow(image_1_corners)
    ax1.set_title('Image 1', fontsize=50)
    ax2.imshow(image_2_corners)
    ax2.set_title('Image 2', fontsize=50)
    plt.show()

def scatter_plot(image_1, image_2, P1, P2, camera_model):
    
    _, corners1 = cv.findChessboardCorners(image_1, (9,6))
    _, corners2 = cv.findChessboardCorners(image_2, (9,6))
    # dst1 = cv.undistortPoints(corners1,camera_model['M1'], camera_model['dist1'])
    # dst2 = cv.undistortPoints(corners2,camera_model['M2'], camera_model['dist2'])
    print(corners1.shape)
    triangulate = cv.triangulatePoints(P1,P2,corners1,corners2)
    x,y,z,w = triangulate
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(*triangulate)
    plt.show()

def save_parameters(camera_model, rect_params):
    np.savetxt("Parameters/rotation_calib.csv",camera_model['R'], delimiter=",")
    np.savetxt("Parameters/translation_calib.csv",camera_model['T'], delimiter=",")
    np.savetxt("Parameters/essential_matrix.csv",camera_model['E'], delimiter=",")
    np.savetxt("Parameters/fundamental_matrix.csv",camera_model['F'], delimiter=",")
    np.savetxt("Parameters/R1_rectify.csv",rect_params['R1'], delimiter=",")
    np.savetxt("Parameters/R2_rectify.csv",rect_params['R2'], delimiter=",")
    np.savetxt("Parameters/P1_rectify.csv",rect_params['P1'], delimiter=",")
    np.savetxt("Parameters/P2_rectify.csv",rect_params['P2'], delimiter=",")
    np.savetxt("Parameters/Q_rectify.csv",rect_params['Q'], delimiter=",")
    



if __name__ == "__main__":
    left_imgs = [cv.imread(file) for file in glob.glob("./images/task_2/left_0.png")]
    right_imgs = [cv.imread(file) for file in glob.glob("./images/task_2/right_0.png")]
    left_img_points, left_obj_points = get_image_and_object_points(left_imgs)
    right_img_points, right_obj_points = get_image_and_object_points(right_imgs)
    left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion = load_camera_parameters()
    camera_model =\
         stereo_calibrate(\
             left_obj_points, left_img_points, right_img_points, left_camera_intrinsics, left_camera_distortion, right_camera_intrinsics, right_camera_distortion)
    rect_params = stereo_rectify(camera_model)
    original_image = cv.imread('./images/task_2/left_0.png')
    original_image2 = cv.imread('./images/task_2/right_0.png')
    undistorted_image = get_undistorted_image(original_image,camera_model['M1'], camera_model['dist1'], rect_params['R1'], rect_params['P1'])
    undistorted_image2 = get_undistorted_image(original_image2,camera_model['M2'], camera_model['dist2'], rect_params['R2'], rect_params['P2'])
    # plot_images(undistorted_image, undistorted_image2)
    save_parameters(camera_model, rect_params)
    scatter_plot( undistorted_image, undistorted_image2, rect_params['P1'], rect_params['P2'], camera_model)
