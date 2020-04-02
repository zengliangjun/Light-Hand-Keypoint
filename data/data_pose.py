from data.data_3d import Data3dWarp
import numpy as np
import random
import cv2
import data.utils as _utils
from torch.utils.data import ConcatDataset

SCALE = 1.25
EPSILON = 1e-6

class DataPose(Data3dWarp):
    def __init__(self, db, config):
        super(DataPose, self).__init__(db)
        self.config = config

    def _getbbox(self, _pts):
        _minx = int(min(_pts[:, 0]))
        _maxx = int(max(_pts[:, 0]))
        _miny = int(min(_pts[:, 1]))
        _maxy = int(max(_pts[:, 1]))
        _centx = (_minx + _maxx) / 2
        _centy = (_miny + _maxy) / 2
        _half_width = (_maxx - _minx) / 2
        _half_height = (_maxy - _miny) / 2
        _half = max(_half_width, _half_height) * SCALE

        _minx = int(_centx - _half)
        _maxx = int(_centx + _half)
        _miny = int(_centy - _half)
        _maxy = int(_centy + _half)
        return (_minx, _miny, _maxx, _maxy)

    def _gettarget(self):
        return (0, 0, self.config.input_size, self.config.input_size)

    def _gen_trans(self, _source, _target, _img):
        _src = np.zeros((3, 2), dtype=np.float32)
        _src[0, :] = np.array(((_source[0] + _source[2]) / 2, (_source[1] + _source[3]) / 2), dtype=np.float32)
        _src[1, :] = np.array(((_source[2], (_source[1] + _source[3]) // 2)), dtype=np.float32)
        _src[2, :] = np.array((((_source[0] + _source[2]) // 2, _source[3])), dtype=np.float32)

        _dst = np.zeros((3, 2), dtype=np.float32)
        _dst[0, :] = np.array(((_target[0] + _target[2]) // 2, (_target[1] + _target[3]) // 2))
        _dst[1, :] = np.array((_target[2], (_target[1] + _target[3]) // 2))
        _dst[2, :] = np.array(((_target[0] + _target[2]) // 2, _target[3]))

        _trans = cv2.getAffineTransform(np.float32(_src), np.float32(_dst))
        _patch = None
        if _img is not None:
            _patch = cv2.warpAffine(_img, _trans, (int(self.config.input_size), int(self.config.input_size)), flags=cv2.INTER_LINEAR)

        return _trans, _patch

    def _get_point2d(self, _skel_2d, _trans):
        _src = np.concatenate([_skel_2d, np.ones((_skel_2d.shape[0], 1), dtype=np.float)], axis=1)
        _des = _trans.dot(_src.T).T
        return _des[:, 0:2]

    def _get_unit_vec(self, _uv2d, _skel3d):
        _vect_size = self.config.input_size // self.config.vector_strid
        _vect_grid = np.zeros(
            (3 * 20, _vect_size, _vect_size),  # exclude instance count
            dtype=np.float32
        )

        _unit_vecs = []
        for i, _link in enumerate(_utils._links()):
            _start = _skel3d[list(_link[:-1]), :]
            _end = _skel3d[list(_link[1:]), :]

            _vector = _end - _start
            _unit_vec = _vector / (np.linalg.norm(_vector, axis=1, keepdims=True) + EPSILON)

            _uv_end = _uv2d[list(_link[1:]), :]
            for _k, _uv in enumerate(_uv_end):
                _ix, _iy = int(_uv[0]), int(_uv[1])
                if 0 <= _iy < _vect_size and 0 <= _ix < _vect_size:
                    _vect_grid[3 * (i * 4 + _k), _iy, _ix] = _unit_vec[_k][0]
                    _vect_grid[3 * (i * 4 + _k) + 1, _iy, _ix] = _unit_vec[_k][1]
                    _vect_grid[3 * (i * 4 + _k) + 2, _iy, _ix] = _unit_vec[_k][2]

                    #print(3 * (i * 4 + _k), 3 * (i * 4 + _k) + 1, 3 * (i * 4 + _k) + 2)
                else:
                    print(_k, _ix, _iy)

        return _vect_grid

    def __getitem__(self, _idx):
        _camera, _img, _skel_2d, _skel3d, _side = super(DataPose, self).__getitem__(_idx)

        _bbox = self._getbbox(_skel_2d)
        _target = self._gettarget()

        _trans, _crop = self._gen_trans(_bbox, _target, _img)

        if random.random() <= 0.5: # flip
            _skel_2d[:, 0] = _img.shape[1] - _skel_2d[:, 0]
            _skel3d[:, 0] = - _skel3d[:, 0]
            _side = 1 - _side

            _bbox = self._getbbox(_skel_2d)
            _trans, _ = self._gen_trans(_bbox, _target, None)
            _crop = _crop[:, ::-1, :]

        _uv2d = self._get_point2d(_skel_2d, _trans)
        _crop_img = _crop
        _crop_img = _crop_img.transpose((2,0,1))
        from chainercv.links.model.ssd import random_distort
        _crop_img = random_distort(_crop_img)

        if False:
            _crop_img_cop = _crop_img.transpose((1,2,0))
            self._vis(_crop_img_cop, _uv2d, _skel3d)

        return np.array(_crop_img, dtype=np.float32), \
            np.array(_uv2d / self.config.pose_strid, dtype=np.float32), \
            np.array(self._get_unit_vec(_uv2d / self.config.vector_strid, _skel3d), dtype=np.float32), \
            np.uint8(_side)

    def _vis(self, _img, _skel_2d, _skel3d):
        from matplotlib import pyplot as plt
        from data.visual import _vis_point, _vis_edge

        _h, _w, _ = _img.shape

        fig = plt.figure(figsize=(8, 8))
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122, projection="3d")

        _vis_point(_skel_2d, img=_img, ax = ax1)
        _vis_edge(_skel_2d, ax = ax1)

        _vis_point(_skel3d, ax = ax2)
        _vis_edge(_skel3d, ax = ax2)

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_zlabel("z")
        ax2.view_init(-65, -90)

        plt.show()

class DataPose2D(DataPose):
    def __init__(self, db, config):
        super(DataPose2D, self).__init__(db, config)

    def __getitem__(self, _idx):
        _camera, _img, _skel_2d, _skel3d, _side = super(DataPose, self).__getitem__(_idx)

        _bbox = self._getbbox(_skel_2d)
        _target = self._gettarget()

        _trans, _crop = self._gen_trans(_bbox, _target, _img)

        if random.random() <= 0.5: # flip
            _skel_2d[:, 0] = _img.shape[1] - _skel_2d[:, 0]
            _skel3d[:, 0] = - _skel3d[:, 0]
            _side = 1 - _side

            _bbox = self._getbbox(_skel_2d)
            _trans, _ = self._gen_trans(_bbox, _target, None)
            _crop = _crop[:, ::-1, :]

        _uv2d = self._get_point2d(_skel_2d, _trans)
        _crop_img = _crop
        _crop_img = _crop_img.transpose((2,0,1))
        from chainercv.links.model.ssd import random_distort
        _crop_img = random_distort(_crop_img)

        if False:
            _crop_img_cop = _crop_img.transpose((1,2,0))
            self._vis(_crop_img_cop, _uv2d, _skel3d)

        return np.array(_crop_img, dtype=np.float32), \
            np.array(_uv2d / self.config.input_size, dtype=np.float32), \
            np.uint8(_side)



class Pose2DDb(ConcatDataset):
    def __init__(self, config, _train = False):
        _db_list = []
        for _db_class in config.db_classes:
            _db = _db_class(_train = _train)
            _db_list.append(DataPose2D(_db, config))

        super(Pose2DDb, self).__init__(_db_list)
