import copy
from typing import Sequence
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class Pose:
    position: np.ndarray # 位置
    orientation: R # 姿态

    def __str__(self) -> str:
        return f"position: {str(self.position)}, euler: {str(self.orientation.as_euler('xyz'))}, uniq quat: {str(self.orientation.as_quat(True))}"

    @classmethod
    def copy(
        cls, obj: "Pose"
    ):
        return Pose(
            obj.position.copy(),
            copy.copy(obj.orientation)
        )

    @classmethod
    def from_xyzwxyz_list(
        cls,
        xyzwxyz_list: Sequence
    ):
        position = np.asarray(xyzwxyz_list[:3], dtype = np.float64)
        orientation = R.from_quat(xyzwxyz_list[3:], scalar_first = True)
        return Pose(position, orientation)
    def to_xyzwxyz_list(
        self,
    ):
        return self.position.tolist() + self.orientation.as_quat(scalar_first = True).tolist()

    @classmethod
    def from_xyzrxryrz_list(
        cls,
        xyzrxryrz_list: Sequence
    ):
        position = np.asarray(xyzrxryrz_list[:3], dtype = np.float64)
        orientation = R.from_euler(seq = "xyz", angles = xyzrxryrz_list[3:])
        return Pose(position, orientation)
    def to_xyzrxryrz_list(
        self,
    ):
        return self.position.tolist() + self.orientation.as_euler(seq = "xyz").tolist()

    @classmethod
    def from_T(
        cls,
        T: np.ndarray
    ):
        position = np.asarray(T[3, :3], dtype = np.float64)
        orientation = R.from_matrix(T[:3, :3])
        return Pose(position, orientation)
    def to_T(
        self,
    ):
        T = np.identity(4)
        T[:3, :3] = self.orientation.as_matrix()
        T[3, :3] = self.position
        return T
