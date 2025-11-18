from typing import Optional, Sequence, Tuple, Union
import numpy as np
from dataclasses import dataclass
import cv2

from .pose_utility import Pose

# 三维点列表以 (3, k) 的矩阵表示, 点坐标按列排列

def is_3d_pts(
    pts: np.ndarray
):
    return (len(pts.shape) == 2) and (pts.shape[0] == 3)

def assert_is_3d_pts(
    pts: np.ndarray
):
    assert is_3d_pts(pts), f"数组形状 {pts.shape} 不是合法的三维点"

def pts_to_homo(
    pts: np.ndarray
):
    '''
    将三维的点坐标转为齐次坐标
    
    @pts: (N,3 or 2) will homogeneliaze the last dimension
    '''
    assert_is_3d_pts(pts)

    homo = np.concat((pts, np.ones((1, pts.shape[1]))), axis = 0)
    return homo

def ptslist_to_pts(
    ptlist: Sequence
):
    return np.stack(
        tuple(np.asarray(pt).reshape(3) for pt in ptlist), axis = 1
    )

def pts_append_point(
    pts: np.ndarray,
    point: Union[np.ndarray, Sequence]
):
    return np.concat((pts, np.asarray(point).reshape((3, 1))), axis = 1)

@dataclass
class CameraVisualize:
    
    cam_k: np.ndarray

    def project_3d_to_2d(
        self,
        pts_ob: Union[np.ndarray, Sequence],
        ob_in_cam: np.ndarray = np.identity(4),
    ):
        '''
        将物体坐标系下的三维点投影到相机平面

        Args:
            pt_ob (np.ndarray): (3, K) 的矩阵, 点按列排列, 也可以传入单个点 (三元素即可)
            ob_in_cam (np.ndarray): 相机坐标系观测下的物体位姿
        '''

        if not isinstance(pts_ob, np.ndarray):
            pts_ob = np.asarray(pts_ob)
        if pts_ob.shape == (3,):
            pts_ob = pts_ob.reshape((3, 1))
        
        assert_is_3d_pts(pts_ob)
        pts_homo_ob = pts_to_homo(pts_ob)

        projected = self.cam_k @ ((ob_in_cam @ pts_homo_ob)[:3, :])
        projected = np.asarray(projected / projected[2])
        return np.asarray(
            np.round(projected[:2, :]), dtype = np.int32
        )

    def draw_line3d(
        self,
        img: np.ndarray,
        ob_in_cam: np.ndarray, 

        start: Union[Sequence, np.ndarray],
        end: Union[Sequence, np.ndarray],
        
        line_color: Tuple[int, int , int] = (0,255,0), 
        line_width: int = 2
    ):
        pts3 = ptslist_to_pts((start, end))
        project = self.project_3d_to_2d(pts3, ob_in_cam)

        img = cv2.line(
            img, 
            project[:, 0].tolist(), 
            project[:, 1].tolist(), 
            color = line_color, 
            thickness = line_width, 
            lineType = cv2.LINE_AA
        )
        return img

    def draw_xyz_axis(
        self,
        img: np.ndarray,
        ob_in_cam: np.ndarray, 

        line_scale: float = 0.1, 
        line_width: int = 2,

        label_is_drwa: bool = True,
        label_size: int = 2,
    ):
        axis_pts = np.identity(3) * line_scale
        axis_pts = pts_append_point(axis_pts, np.zeros(3))
        axis_project = self.project_3d_to_2d(axis_pts, ob_in_cam)

        axis_name = ("x", "y", "z")

        for i in range(3):
            color = [0, 0, 0]
            color[i] = 255

            img = cv2.line(
                img, 
                axis_project[:, i].tolist(), 
                axis_project[:, -1].tolist(), 
                color = color, 
                thickness = line_width, 
                lineType = cv2.LINE_AA
            )

            if label_is_drwa:
                img = cv2.putText(
                    img, axis_name[i], axis_project[:, i].tolist(), 
                    cv2.FONT_HERSHEY_PLAIN, label_size, color, line_width, cv2.LINE_AA
                )
        return img

    def draw_bbox(
        self,
        img: np.ndarray, 
        ob_in_cam: np.ndarray, 

        bbox_min_xyz: np.ndarray, 
        bbox_max_xyz: np.ndarray, 

        line_color: Tuple = (0,255,0), 
        line_width: int = 2
    ):
        xmin, ymin, zmin = bbox_min_xyz
        xmax, ymax, zmax = bbox_max_xyz

        for y in [ymin,ymax]:
            for z in [zmin,zmax]:
                start = np.array([xmin,y,z])
                end = start + np.array([xmax-xmin,0,0])

                img = self.draw_line3d(
                    img, ob_in_cam, start, end, line_color, line_width
                )

        for x in [xmin,xmax]:
            for z in [zmin,zmax]:
                start = np.array([x,ymin,z])
                end = start+np.array([0,ymax-ymin,0])
    
                img = self.draw_line3d(
                    img, ob_in_cam, start, end, line_color, line_width
                )
    
        for x in [xmin,xmax]:
            for y in [ymin,ymax]:
                start = np.array([x,y,zmin])
                end = start+np.array([0,0,zmax-zmin])

                img = self.draw_line3d(
                    img, ob_in_cam, start, end, line_color, line_width
                )

        return img

    def draw_gripper(
        self,
        img: np.ndarray, 
        ob_in_cam: np.ndarray, 

        gripper_width: float = 0.050,
        gripper_depth: float = 0.050,
        gripper_tail: float = 0.050,

        line_width: int = 2,
        label: Optional[str] = None,
        label_size: int = 2,

    ):
        '''
        绘制夹爪, 规定夹爪坐标系
        - x 轴为法兰盘指向夹爪开口方向
        - y 轴为夹爪张开方向, 初始状态下朝基座看在指向左侧
        - z 轴初始状态下指向机器人基座
        - 坐标系原点为夹爪 x 轴方向最大的有效开口中心

        Args:
            img (np.ndarray): 图片
            ob_in_cam (np.ndarray): 相机坐标系观察下的夹爪位姿
            gripper_width (float): 夹爪半张开宽度, 仅视觉效果
            gripper_depth (float): 夹爪有效开口深度, 仅视觉效果
            gripper_tail (float): 到法兰盘距离, 仅视觉效果
            line_width (int): 图例线宽

        '''

        left_fingger = [
            [0, -gripper_width, 0], [-gripper_depth, -gripper_width, 0]
        ]
        right_fingger = [
            [0, gripper_width, 0], [-gripper_depth, gripper_width, 0]
        ]
        gripper_botton = [
            [-gripper_depth, -gripper_width, 0], [-gripper_depth, gripper_width, 0]
        ]
        gripper_tailrod = [
            [-gripper_depth, 0, 0], [-(gripper_depth + gripper_tail), 0, 0]
        ]

        img = self.draw_line3d(
            img, ob_in_cam, left_fingger[0], left_fingger[1], (255, 0, 0), line_width
        )
        img = self.draw_line3d(
            img, ob_in_cam, right_fingger[0], right_fingger[1], (255, 0, 0), line_width
        )
        img = self.draw_line3d(
            img, ob_in_cam, gripper_botton[0], gripper_botton[1], (0, 255, 0), line_width
        )
        img = self.draw_line3d(
            img, ob_in_cam, gripper_tailrod[0], gripper_tailrod[1], (0, 0, 255), line_width
        )

        if label is not None:
            axis_project = self.project_3d_to_2d(gripper_tailrod[1], ob_in_cam)
            img = cv2.putText(
                img, label, axis_project[:, 0].tolist(), 
                cv2.FONT_HERSHEY_PLAIN, label_size, (0, 0, 0), line_width, cv2.LINE_AA
            )

        return img
    
if __name__ == "__main__":
    pts = ptslist_to_pts(([0, 2, 3], [3, 5, 6]))
    print(f"make pts:\n{pts}")

    pts = pts_append_point(pts, [6, 8, 9])
    print(f"make pts:\n{pts}")

    pts4 = pts_to_homo(pts)
    print(f"pts in homo:\n{pts4}")

    ob_in_cam = Pose.from_xyzrxryrz_list([1, 0, 0, 0, 0, 0]).to_T()
    print(f"move alone x with 1:\n {(ob_in_cam @ pts4)[:3, :]}")

    cam = CameraVisualize(
        np.identity(3)
    )
    project = cam.project_3d_to_2d(pts)
    print(f"f=1 project:\n{project}")