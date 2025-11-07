import copy
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass

@dataclass
class Pose:
    position: np.ndarray # 位置
    orientation: R # 姿态
    _T: Optional[np.ndarray] = None # 缓存的齐次矩阵, 用于姿态相关运算

    @classmethod
    def copy(
        cls, obj: "Pose"
    ):
        return Pose(
            obj.position.copy(),
            copy.copy(obj.orientation)
        )
    def calc_T(
        self,
    ):
        if self._T is None:
            T = np.identity(4)
            T[:3, :3] = self.orientation.as_matrix()
            T[:3, 3] = self.position
            self._T = T
        
        return self._T
    def inv(
        self,
    ):
        return self.from_T(np.linalg.inv(self.calc_T()))
    @classmethod
    def pose_mul(
        cls,
        a: "Pose",
        b: "Pose" 
    ):
        return np.linalg.matmul(a.calc_T(), b.calc_T())
    @classmethod
    def distance(
        cls,
        a: "Pose",
        b: "Pose", 
    ) -> Tuple[float, float]:
        '''
        获取两个位姿的距离差 (标量单位) 与角度差 (单位 rad)
        '''
        diff_mat = a * b.inv()
        pos_diff = np.linalg.norm(diff_mat.position, ord = 2)
        rot_diff = np.linalg.norm(diff_mat.orientation.as_rotvec(degrees = False), ord = 2)

        return (float(pos_diff), float(rot_diff))

    def __mul__(
        self,
        other: "Pose"   
    ):
        return self.from_T(self.pose_mul(self, other))
    def __str__(self) -> str:
        return f"position: {str(self.position)}, euler: {str(self.orientation.as_euler('xyz'))}, uniq quat: {str(self.orientation.as_quat(True))}"

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
        orientation = R.from_euler(seq = "xyz", angles = np.asarray(xyzrxryrz_list[3:], dtype = np.float64))
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
        position = np.asarray(T[:3, 3], dtype = np.float64)
        orientation = R.from_matrix(T[:3, :3])
        return Pose(position, orientation, _T = T)
    def to_T(
        self,
    ):
        return self.calc_T()
