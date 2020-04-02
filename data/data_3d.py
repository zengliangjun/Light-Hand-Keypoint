import numpy as np
import cv2
from torch.utils.data import Dataset

_angle_range = [-45, 45]
_list_range = [i for i in range(_angle_range[0], _angle_range[1])]
class Data3dWarp(Dataset):
    def __init__(self, db):
        super(Data3dWarp, self).__init__()
        self.db = db

    def __len__(self):
        return len(self.db)

    def _angle(self):
        np.random.seed()
        _angle = np.random.choice(_list_range)
        #print(_angle)
        return _angle

    def _rotate(self, _angle, _skel_2d):
        import math
        _angle = np.deg2rad(_angle)
        _rmat_uv = np.array([
            [math.cos(_angle), -math.sin(_angle)],
            [math.sin(_angle), math.cos(_angle)],
        ], dtype=np.float)

        return _rmat_uv.dot(_skel_2d.transpose()).transpose()

    def _uv2d(self, _camera, _skel3d):
        _skel_hom2d = _camera.dot(_skel3d.transpose()).transpose()
        _skel_2d = (_skel_hom2d / _skel_hom2d[:, 2:])[:, :2]
        return _skel_2d

    def _camera2one(self, _camera):
        return np.array([
            [_camera[0][0] / _camera[0][2], 0, 1],
            [0, _camera[1][1] / _camera[1][2], 1],
            [0, 0, 1]
        ], dtype=np.float)

    def _calcute3d(self, _2d, _3dz, _camera2one):
        _back_xyz = np.array([
            [1 / _camera2one[0][0], 0, - _camera2one[0][2] / _camera2one[0][0]],
            [0, 1 / _camera2one[1][1], - _camera2one[1][2] / _camera2one[1][1]],
            [0, 0, 1]
        ], dtype=np.float)

        _n = _3dz.shape[0]
        _uvz = np.concatenate([_2d, np.ones((_n, 1), dtype=np.float)], axis=1)
        _xyz = _back_xyz.dot(_uvz.transpose()).transpose()
        return _xyz * _3dz

    def _rotate_img(self, image, angle):
        (h, w) = image.shape[:2]

        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, angle, 1)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated

    def __getitem__(self, _idx):
        _camera, _skel3d, _side = self.db[_idx]
        _img = self.db.getimg(_idx)[:, :, ::-1]
        _angle = self._angle()

        _cent_uv = np.array((_img.shape[:2])[::-1], dtype=np.float).reshape(1, 2) / 2
        _skel_2d = self._uv2d(_camera, _skel3d)

        _tmp_2d = _skel_2d - _cent_uv
        _tmp_2d = self._rotate(_angle, _tmp_2d)
        _rotate_skel_2d = _tmp_2d + _cent_uv
        _rotate_skel3d = self._calcute3d(_rotate_skel_2d, _skel3d[:, 2:], _camera)

        _rotate_img = self._rotate_img(_img, - _angle)
        if False:
            self._vis(_img, _skel_2d.copy(), _skel3d, _rotate_img, _rotate_skel_2d.copy(), _rotate_skel3d)

        if _side == 'left':
            _side = 0
        else:
            _side = 1

        return _camera, _rotate_img, _rotate_skel_2d, _rotate_skel3d, _side

    def _vis(self, _img, _skel_2d, _skel3d, _rotate_img, _rotate_skel_2d, _rotate_skel3d):
        from matplotlib import pyplot as plt
        from data.visual import _vis_point, _vis_edge

        _h, _w, _ = _img.shape
        _h2, _w2, _ = _rotate_img.shape

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223, projection="3d")
        ax4 = fig.add_subplot(224, projection="3d")

        _vis_point(_skel_2d, img=_img, ax = ax1)
        _vis_edge(_skel_2d, ax = ax1)

        _vis_point(_rotate_skel_2d, img=_rotate_img, ax = ax2)
        _vis_edge(_rotate_skel_2d, ax = ax2)

        _vis_point(_skel3d, ax = ax3)
        _vis_edge(_skel3d, ax = ax3)
        
        _vis_point(_rotate_skel3d, ax = ax4)
        _vis_edge(_rotate_skel3d, ax = ax4)

        for ax in [ax3, ax4]:
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
            ax.view_init(-65, -90)
        
        plt.show()


if __name__ == '__main__':
    from data.ho3d.ho3d import Ho3d
    _db = Ho3d()

    _3d_warp = Data3dWarp(_db)
    for _idx in range(len(_3d_warp)):
        _tmp = _3d_warp[_idx]