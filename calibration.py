import glob
import os

import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt


def get_chessboard_points(chessboard_shape, dx, dy):
    xcoords = np.arange(0, chessboard_shape[1] * dx, dx)
    ycoords = np.arange(0, chessboard_shape[0] * dy, dy)
    coords_2d = np.array(np.meshgrid(xcoords, ycoords)).T.reshape(chessboard_shape[0] * chessboard_shape[1], 2)
    coords_3d = np.insert(coords_2d, 2, 0, axis=1)
    return coords_3d


def draw_corners(image: np.ndarray, corners):
    image = cv2.drawChessboardCorners(image, (9, 6), corners, 1)
    return image


def get_corners(images: list) -> list:
    corners = [None] * len(images)

    #  Calibration template size
    c_size = (9, 6)

    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    window_size = (11, 11)
    zero_zone = (-1, -1)

    for i, image in enumerate(images):
        # Transform to gray and get chessboard corners
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, chessboard_corners = cv2.findChessboardCorners(gray_image, c_size, None)

        if ret:
            #  Refine corner estimation
            cv2.cornerSubPix(gray_image, chessboard_corners, window_size, zero_zone, criteria)
            corners[i] = chessboard_corners
        else:
            print(f"Could not find corners in {filenames[i]}")

    return corners


def load_images(filenames):
    images = []

    for i, filename in enumerate(filenames):
        image = cv2.imread(filename)
        images.append(image)

    return images, filenames


def calibrate_camera(corners, cb_points, image_shape):
    # Extract the list of valid images with all corners
    valid_corners = [i for i, elem in enumerate(corners) if elem is not None]
    num_valid_images = len(valid_corners)

    # Prepare input data
    # object_points: numpy array with dimensions (number_of_images, number_of_points, 3)
    object_points = np.repeat(np.expand_dims(cb_points, axis=0), num_valid_images, axis=0).astype(np.float32)
    # print(object_points.shape)

    # image_points: numpy array with dimensions (number_of_images, number_of_points, 2)
    image_points = np.array(corners, np.float32)[:, :, 0, :]
    # print(image_points.shape)

    # Calibrate for square pixels corners standard
    rms, intrinsics, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_shape,
                                                                     None, None, flags=cv2.CALIB_FIX_ASPECT_RATIO)

    return rms, intrinsics, dist_coeffs, rvecs, tvecs


def get_camera_calibration():
    orig_folder = "out/calibration"
    filenames = sorted(glob.glob(os.path.join(orig_folder, "*.png")))
    images, filenames = load_images(filenames)
    corners = get_corners(images)

    # Ancho del cuadro en la plantilla de calibración, 24mm
    cb_points = get_chessboard_points((9, 6), 24, 24)

    rms, intrinsics, dist_coeffs, rvecs, tvecs = calibrate_camera(corners, cb_points, images[0].shape[:-1])
    return rms, intrinsics, dist_coeffs, rvecs, tvecs


if __name__ == "__main__":
    rms, intrinsics, dist_coeffs, rvecs, tvecs = get_camera_calibration()
    print(intrinsics)
    print(rms)

#    for i in range(10):
#        im = draw_corners(images[i], corners[i])
#        plt.imshow(im)
#        plt.show()

