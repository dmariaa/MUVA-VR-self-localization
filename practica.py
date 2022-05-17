from typing import Union

import cv2.cv2 as cv2

import matplotlib
import matplotlib.pyplot as plt
from pupil_apriltags import Detector, Detection

from video_player import VideoPlayer
from calibration import get_camera_calibration
from tools import *

matplotlib.use('Qt5Agg')


def calculate_corners(K: np.ndarray):
    K_inv = np.linalg.inv(K)
    cx = int(K[0, 2])
    cy = int(K[1, 2])

    corners = K_inv @ np.array([
        [0, 0, 1],
        [cx * 2, 0, 1],
        [cx * 2, cy * 2, 1],
        [0, cy * 2, 1],
    ]).T
    corners = corners.T
    corners = np.append(corners, np.array([[1, 1, 1, 1]]).T, axis=1)
    return corners


def show_tag(frame: np.ndarray, tag: Detection, k: np.ndarray, r: np.ndarray, t: np.ndarray):
    id = tag.tag_id
    center = tag.center
    corners = tag.corners

    # center
    frame = cv2.drawMarker(frame, position=[int(center[0]), int(center[1])], color=colors[4], thickness=2)
    frame = cv2.putText(frame, text=f"TAG ID: {id}", org=[int(center[0]) - 30, int(center[1]) - 30], color=colors[4],
                        fontScale=0.5, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

    for i, c in enumerate(corners):
        # point 0 -> point 1
        frame = cv2.putText(frame, text=f"{i}", org=[int(c[0] + ref_marker_points[i, 0] * 20),
                                                     int(c[1] + ref_marker_points[i, 1] * 20)],
                            color=colors[0], fontScale=0.5, thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)

        frame = cv2.drawMarker(frame, position=[int(c[0]), int(c[1])], color=colors[0], thickness=2)

        frame = cv2.line(frame,
                         pt1=[int(c[0]), int(c[1])],
                         pt2=[int(corners[(i + 1) % 4, 0]), int(corners[(i + 1) % 4, 1])],
                         color=colors[i],
                         thickness=2)

    frame = cv2.drawFrameAxes(frame, k, dist_coeffs, r, t, 0.75, 10)

    return frame


def detect_tags(frame: np.ndarray):
    tags = at_detector.detect(frame, estimate_tag_pose=False, camera_params=None, tag_size=None)

    if len(tags) > 0:
        for tag in tags:
            image_points = np.append(tag.corners, [tag.center], axis=0)
            res, rvec_tag, tvec_tag = cv2.solvePnP(ref_marker_points2, image_points, intrinsics, dist_coeffs,
                                                   flags=cv2.SOLVEPNP_IPPE)

            rvec_tag, tvec_tag = cv2.solvePnPRefineLM(ref_marker_points2, image_points, intrinsics, dist_coeffs, rvec_tag, tvec_tag)

            rvec_tag = cv2.Rodrigues(rvec_tag)[0]
            yield tag, [rvec_tag, tvec_tag]


def cam_self_projection(tags: list, tag_mats: list):
    '''
    Calculates camera self projection for multiple tags
    :param tag_mats:
    :type tag_mats:
    :return:
    :rtype:
    '''
    cam_center = np.zeros(3)
    corners = np.zeros((4, 3))

    for i, tag_mat in enumerate(tag_mats):
        r, t = tag_mat
        RT_inv = np.append(np.append(r.T, -r.T @ t, axis=1), [[0, 0, 0, 1]], axis=0)
        tag_matrix = tags_to_world[tags[i].tag_id]

        p = tag_matrix @ RT_inv @ [[0], [0], [0], [1]]
        cam_center += (p[:-1] / p[-1]).flatten()

        c = tag_matrix @ RT_inv @ corner_points.T
        c = c[:-1, :] / c[-1, :]
        corners += c.T

    cam_center /= len(tag_mats)
    corners /= len(tag_mats)

    return cam_center, corners


class CameraPlotter:
    def __init__(self, K: np.ndarray, cam_size: float = 1.):
        self.K = K

        self.cam_size = cam_size
        self.figure = plt.figure(figsize=(6.4, 6.4))

        # self.ax_image = plt.subplot2grid((1, 3), (0, 0), colspan=2)
        # self.ax_image.set_axis_off()

        self.view_azimut = -63
        self.view_elevation = -157
        self.ax = plt.subplot2grid((1, 1), (0, 0), projection='3d')
        self.ax.set_xlim3d([-1., 10.])
        self.ax.set_ylim3d([-1., 10.])
        self.ax.set_zlim3d([-10., 1.])
        self.ax.view_init(elev=self.view_elevation, azim=self.view_azimut)

        self._draw_world_axis()

        self.line, = self.ax.plot3D([], [], [])

        plt.show(block=False)
        plt.pause(0.001)

        self.center_points = np.empty((0, 3), dtype=float)

    def _draw_world_axis(self):
        self.ax.plot3D([0, 1], [0, 0], [0, 0], c='red')
        self.ax.plot3D([0, 0], [0, 1], [0, 0], c='green')
        self.ax.plot3D([0, 0], [0, 0], [0, 1], c='blue')

    def _draw_template(self, tag_matrix):
        t = tag_matrix @ ref_marker_points.T
        markers = np.append(t[:, 0:4], t[:, 0:1], axis=1)

        self.ax.plot3D(markers[0, :], markers[1, :], markers[2, :], c='blue')
        self.ax.scatter3D(t[0, 0:4], t[1, 0:4], t[2, 0:4], s=4., c='red')
        self.ax.scatter3D(t[0, 4], t[1, 4], t[2, 4], s=4., c='green')

        a = tag_matrix @ ref_marker_axis.T
        self.ax.plot3D(a[0, 0:2:1], a[1, 0:2:1], a[2, 0:2:1], c='red')
        self.ax.plot3D(a[0, 0:3:2], a[1, 0:3:2], a[2, 0:3:2], c='green')
        self.ax.plot3D(a[0, 0:4:3], a[1, 0:4:3], a[2, 0:4:3], c='blue')

    def _draw_camera(self, cam_center: np.ndarray, corners: np.ndarray):
        self.center_points = np.append(self.center_points[-20:], [cam_center], axis=0)

        for i, v in enumerate(corners):
            v2 = corners[(i + 1) % 4]
            self.ax.plot3D([v[0], v2[0]], [v[1], v2[1]], [v[2], v2[2]], c='black')
            self.ax.plot3D([cam_center[0], v[0]], [cam_center[1], v[1]], [cam_center[2], v[2]], c='black')

        self.ax.plot3D(self.center_points[:, 0], self.center_points[:, 1], self.center_points[:, 2], c='black')

    def draw(self, cam_center: np.ndarray, corners: np.ndarray, tags):
        self.ax.clear()
        self.ax.set_xlim3d([-1., 10.])
        self.ax.set_ylim3d([-1., 10.])
        self.ax.set_zlim3d([-10., 1.])

        self._draw_world_axis()

        for tag in tags:
            self._draw_template(tags_to_world[tag.tag_id])

        self._draw_camera(cam_center, corners)


class FrameProcessor(VideoPlayer):
    def __init__(self, intrinsics: np.ndarray, dist_coeffs: np.ndarray, **kwargs):
        super(FrameProcessor, self).__init__(**kwargs)

        self.K = intrinsics
        self.dist_coeffs = dist_coeffs
        self.camera_plotter = CameraPlotter(intrinsics)

    def _process_frame(self, frame: np.ndarray) -> (Union[np.ndarray, None], bool):
        # frame = cv2.resize(frame, (640, 480))
        bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        show_plot3d = True
        plot_freq = 1

        tags = []
        tags_mats = []
        for tag, tag_mat in detect_tags(bw_frame):
            R, T = tag_mat
            frame = show_tag(frame, tag, intrinsics, R, T)
            tags.append(tag)
            tags_mats.append(tag_mat)

        if show_plot3d and len(tags) > 0 and self.frame_counter % plot_freq == 0:
            camera_center, corners = cam_self_projection(tags, tags_mats)
            self.camera_plotter.draw(camera_center, corners, tags)

        # self.camera_plotter.ax_image.clear()
        # self.camera_plotter.ax_image.set_axis_off()
        # self.camera_plotter.ax_image.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)

        return cv2.resize(frame, (640, 480)), True


if __name__ == "__main__":
    import argparse


    def options():
        parser = argparse.ArgumentParser()

        parser.add_argument("-i", "--input",
                            help="file name (string) or camera number (int), default camera 0",
                            required=True,
                            default=0)

        parser.add_argument("-c", "--calibration-folder",
                            help="camera calibration files folder",
                            required=True)

        return parser


    args = options().parse_args()
    input_device = args.input
    calib_folder = args.calibration_folder

    at_detector = Detector(families='tag36h11',
                           nthreads=1,
                           quad_decimate=1.0,
                           quad_sigma=0.0,
                           refine_edges=1,
                           decode_sharpening=0.25,
                           debug=0)

    print("Calibrating camera...")
    rms, intrinsics, dist_coeffs, rvecs, tvecs = get_camera_calibration(calib_folder)
    print("Camera calibration done!!!")

    # corners needed to show camera cone
    corner_points = calculate_corners(intrinsics)
    frame_processor = FrameProcessor(intrinsics, dist_coeffs, input=input_device)
    frame_processor.play()
