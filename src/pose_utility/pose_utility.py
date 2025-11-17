import copy
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.spatial.transform import Rotation as R
from dataclasses import field, dataclass

# def is_3d_pts(
#     pts: np.ndarray
# ):
#     return (len(pts.shape) == 2) and (pts.shape[0] == 3)

# def assert_3d_pts(
#     pts: np.ndarray
# ):
#     assert is_3d_pts(pts), f"数组形状 {pts.shape} 不是合法的三维点"

# def pts_to_homo(
#         pts: np.ndarray
#     ):
#     '''
#     将三维的点坐标转为齐次坐标
    
#     @pts: (N,3 or 2) will homogeneliaze the last dimension
#     '''
#     assert_3d_pts(pts)

#     homo = np.concat((pts, np.ones((1, pts.shape[1]))), axis = 0)
#     return homo

@dataclass
class Pose:
    position: np.ndarray # 位置
    orientation: R # 姿态

    _T: Optional[np.ndarray] = None # 缓存的齐次矩阵, 用于姿态相关运算
    _Pose_inv: Optional["Pose"] = None # 缓存的齐次矩阵的, 用于姿态相关运算

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
        if self._Pose_inv is None:
            T_origin = self.calc_T()
            R_inv = T_origin[:3, :3].T
            p_inv = - np.linalg.matmul(R_inv, T_origin[:3, 3])
            
            T_inv = np.identity(4)
            T_inv[:3, :3] = R_inv
            T_inv[:3, 3] = p_inv

            self._Pose_inv = self.from_T(T_inv)
            self._Pose_inv._Pose_inv = self
        return self._Pose_inv

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
        if isinstance(other, Pose):
            return self.from_T(self.pose_mul(self, other))
        else:
            raise RuntimeError(f"{type(other)} is not Pose")

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
        xyzrxryrz_list: Sequence,
        is_degree: bool = False
    ):
        position = np.asarray(xyzrxryrz_list[:3], dtype = np.float64)
        orientation = R.from_euler(seq = "xyz", angles = np.asarray(xyzrxryrz_list[3:], dtype = np.float64), degrees = is_degree)
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
        # Scipy: if the input is not orthogonal, an approximation is created by orthogonalizing the input matrix 
        orientation = R.from_matrix(T[:3, :3])
        return Pose(position, orientation, _T = T)
    def to_T(
        self,
    ):
        return self.calc_T()
    
    @classmethod
    def identity(
        cls
    ):
        return cls.from_T(np.identity(4))

    @dataclass
    class Cfg:
        pose_type: str
        pose_value: Tuple
        pose_args: Dict[str, Union[str, float, bool]] = field(default_factory = lambda: {})

    @classmethod
    def from_cfg(
        cls,
        cfg: Cfg
    ):
        if cfg.pose_type == "xyzwxyz":
            return cls.from_xyzwxyz_list(cfg.pose_value)
        
        elif cfg.pose_type == "xyzrxryrz":
            is_degree = cfg.pose_args.get("is_degree", False)
            assert isinstance(is_degree, bool), f"Pose config args of xyzrxryrz: is_degree should be bool, not {type(is_degree)}"
            return cls.from_xyzrxryrz_list(cfg.pose_value, is_degree)
        
        elif cfg.pose_type == "T":
            return cls.from_T(np.asarray(cfg.pose_value).reshape((4, 4)))
        
        else:
            raise RuntimeError(f"Unknown pose type: {cfg.pose_type}")
