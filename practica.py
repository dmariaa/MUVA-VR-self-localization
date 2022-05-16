import cv2.cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pupil_apriltags import Detector, Detection

from calibration import get_camera_calibration

colors = [
    (255, 255, 0),
    (0, 0, 255),
    (0, 255, 0),
    (255, 0, 0),
    (0, 255, 255)
]

ref_marker_points = np.array([
    [-1, 1, 0],
    [1, 1, 0],
    [1, -1, 0],
    [-1, -1, 0],
    [0, 0, 0]
], dtype=float)

ref_marker_axis = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)

margins = np.array([
    [-1, -1],
    [1, -1],
    [1, 1],
    [-1, 1]
]) * 20


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

    center = k @ (r @ [0, 0, 0] + t.T).T
    center = center[:-1] / center[-1]

    for i in [1, 2, 3]:
        axis = ref_marker_axis[i]
        p = k @ (-r @ (axis * .5) + t.T).T
        p = p[:-1] / p[-1]
        frame = cv2.line(frame, pt1=center.flatten().astype(int), pt2=p.flatten().astype(int),
                         color=colors[i], thickness=8)

    return frame


def detect_tags(frame: np.ndarray):
    tags = at_detector.detect(frame, estimate_tag_pose=False, camera_params=None, tag_size=None)

    if len(tags) > 0:
        for tag in tags:
            image_points = np.append(tag.corners, [tag.center], axis=0)
            res, rvec_tag, tvec_tag = cv2.solvePnP(ref_marker_points, image_points, intrinsics, dist_coeffs,
                                                   flags=cv2.SOLVEPNP_IPPE)
            rvec_tag = cv2.Rodrigues(rvec_tag)[0]
            yield tag, [rvec_tag, tvec_tag]


class CameraPlotter:
    center_point = np.array([0, 0, 0, 1]).T

    def __init__(self, K: np.ndarray, cam_size: float = 1.):
        self.K = K
        self.K_inv = np.linalg.inv(K)
        self.cx = int(K[0, 2])
        self.cy = int(K[1, 2])

        corners = self.K_inv @ np.array([
            [0, 0, 1],
            [self.cx * 2, 0, 1],
            [self.cx * 2, self.cy * 2, 1],
            [0, self.cy * 2, 1],
        ]).T
        corners = corners.T
        self.corners = np.append(corners, np.array([[1, 1, 1, 1]]).T, axis=1)

        self.cam_size = cam_size
        self.figure = plt.figure()

        self.ax = Axes3D(self.figure)
        # self.ax.autoscale(enable=True, axis='both', tight=True)
        self._draw_template()

        self.line, = self.ax.plot3D([], [], [])
        plt.show(block=False)

        self.center_points = np.empty((0, 3), dtype=float)

    def _draw_template(self):
        markers = np.append(ref_marker_points[:-1, :], ref_marker_points[0:1, :], axis=0)

        self.ax.set_xlim3d([-5., 5.])
        self.ax.set_ylim3d([-5., 5.])
        self.ax.set_zlim3d([-5., 5.])

        self.ax.plot3D(markers[:, 0], markers[:, 1], markers[:, 2])
        self.ax.scatter3D(ref_marker_points[:-1, 0], ref_marker_points[:-1, 1], ref_marker_points[:-1, 2], s=4., c='red')
        self.ax.scatter3D(ref_marker_points[-1, 0], ref_marker_points[-1, 1], ref_marker_points[-1, 2], s=4., c='green')

        self.ax.plot3D(ref_marker_axis[0:2:1, 0], ref_marker_axis[0:2:1, 1], ref_marker_axis[0:2:1, 2], c='red')
        self.ax.plot3D(ref_marker_axis[0:3:2, 0], ref_marker_axis[0:3:2, 1], ref_marker_axis[0:3:2, 2], c='green')
        self.ax.plot3D(ref_marker_axis[0:4:3, 0], ref_marker_axis[0:4:3, 1], ref_marker_axis[0:4:3, 2], c='blue')

    def _draw_camera(self, RT):
        cam_center = RT @ self.center_point
        cam_center = cam_center[:-1] / cam_center[-1]

        self.center_points = np.append(self.center_points, [cam_center], axis=0)

        c = (RT @ self.corners.T).T
        c = c[:, :-1] / c[:, -1][:, np.newaxis]

        for i, v in enumerate(c):
            v2 = c[(i + 1) % 4]
            self.ax.plot3D([v[0], v2[0]], [v[1], v2[1]], [v[2], v2[2]], c='black')
            self.ax.plot3D([cam_center[0], v[0]], [cam_center[1], v[1]], [cam_center[2], v[2]], c='black')

        self.ax.plot3D(self.center_points[:, 0], self.center_points[:, 1], self.center_points[:, 2], c='black')

    def draw(self, R: np.ndarray, T: np.ndarray):
        RT = np.append(np.append(-R, T, axis=1), [[0, 0, 0, 1]], axis=0)
        RT_inv = np.linalg.inv(RT)

        self.ax.clear()
        self._draw_template()
        self._draw_camera(RT_inv)

        plt.draw()
        plt.show(block=False)


at_detector = Detector(families='tag36h11',
                       nthreads=1,
                       quad_decimate=1.0,
                       quad_sigma=0.0,
                       refine_edges=1,
                       decode_sharpening=0.25,
                       debug=0)

print("Calibrating camera...")
rms, intrinsics, dist_coeffs, rvecs, tvecs = get_camera_calibration()
print("Camera calibration done!!!")

camera_plotter = CameraPlotter(intrinsics)

capture = cv2.VideoCapture(0)
save_frame = False
show_plot3d = True
plot_freq = 5

cv2.namedWindow('camara', cv2.WINDOW_NORMAL)
cv2.resizeWindow('camara',  60, 40)

i = 0
while capture.isOpened():
    ret, frame = capture.read()
    bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for tag, tag_mat in detect_tags(bw_frame):
        R, T = tag_mat
        frame = show_tag(frame, tag, intrinsics, R, T)

        if show_plot3d and i % plot_freq == 0:
            camera_plotter.draw(R, T)

    frame_to_show = cv2.resize(frame, (640, 480))
    cv2.imshow('camara', frame_to_show)
    if cv2.waitKey(1) == ord('q'):
        plt.show(block=True)
        break

    i += 1

capture.release()
cv2.destroyWindow()
