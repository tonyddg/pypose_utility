from .pose_utility import Pose
from .camera_visualize import CameraVisualize

if __name__ == "__main__":
    import numpy as np

    T0 = Pose.from_T(np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]))

    print(T0 * Pose.from_xyzrxryrz_list([0, 0, 1, 0, 0, 0]))
