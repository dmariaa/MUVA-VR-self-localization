from typing import Union

import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, input: Union[str, int]):
        self.input = input
        self.vc = cv2.VideoCapture(input)
        self.output_window_name = f"Webcam {input}" if type(input) is int else f"{input}"
        self.frame_counter = -1

    def _process_frame(self, frame: np.ndarray) -> (Union[np.ndarray, None], bool):
        return frame, True

    def play(self):
        self.frame_counter = 0
        window_initialized = False

        try:
            while self.vc.isOpened():
                ret, frame = self.vc.read()

                if ret:
                    frame, show = self._process_frame(frame)

                    if show:
                        if not window_initialized:
                            cv2.namedWindow(self.output_window_name)
                            window_initialized = True

                        cv2.imshow(self.output_window_name, frame)

                    if cv2.waitKey(1) == ord('q'):
                        print(f"q pressed")
                        break

                    self.frame_counter += 1
        except KeyboardInterrupt:
            print(f"Cancelled by keyboard interruption")

        self.vc.release()

        if window_initialized:
            cv2.destroyWindow(self.output_window_name)


if __name__ == "__main__":
    import argparse


    def options():
        parser = argparse.ArgumentParser()

        parser.add_argument("-i", "--input",
                            help="file name (string) or camera number (int), default camera 0",
                            required=True,
                            default=0)
        return parser


    args = options().parse_args()
    input_file = args.input

    player = VideoPlayer(input_file)
    player.play()
