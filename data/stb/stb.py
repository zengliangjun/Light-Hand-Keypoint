import os
import os.path as osp
import random
import time

import cv2
import numpy as np
import scipy.io as io
import glob
import data.utils as _utils

class StbBD(object):

    def __init__(self, _root_dir = "/work.data5/STB"):
        random.seed(time.time())
        self.root_dir = os.path.abspath(_root_dir)
        self._setup_cam()
        self._setup()

    def _setup(self):
        '''
        _lable_file = os.path.join(self.root_dir, "labels", "%s_BB.mat" % (self.seq))
        _mat = io.loadmat(_lable_file)

        self.bb_hands = _mat['handPara']
        '''
        self.seq_frames = []
        self.count_frames = 0
        _path = osp.join(self.root_dir, "labels", "*_SK.mat")
        _lable_files = glob.glob(_path)
        _lable_files.sort()
        for _lable_file in _lable_files:
            _seq_name = osp.basename(_lable_file).split('.')[0].split('_')[0]
            _handPara = io.loadmat(_lable_file)['handPara']
            _handPara = np.array(_handPara)
            _hand_poses = np.transpose(_handPara, (2, 1, 0))

            self.seq_frames.append([self.count_frames, _seq_name, _hand_poses])
            self.count_frames += len(_hand_poses)

            #break

    def __len__(self):
        return self.count_frames

    def _id_order(self):
        return np.array([0,
                         17, 18, 19, 20,
                         13, 14, 15, 16,
                         9, 10, 11, 12,
                         5,  6,  7,  8,
                         1,  2,  3,  4])

    def _setup_cam(self):
        _skrotation = np.array([-0.00531, 0.01196, -0.00301], np.float32)
        skrotate2color = cv2.Rodrigues(_skrotation)[0]

        camsk_color_pos = np.array((29.0381,   0.4563,   1.2326), np.float32)
        self.color_translate = skrotate2color.dot(camsk_color_pos)

    def __getitem__(self, _idx):
        for [_count_frames, _seq_name, _hand_poses] in self.seq_frames:
            if _idx >= _count_frames and _idx < (_count_frames + len(_hand_poses)):
                break
            continue

        _idx = _idx - _count_frames        
        _hand_pos = _hand_poses[_idx][self._id_order()]
        _hand_pos[0] = 2 * _hand_pos[0] -  _hand_pos[9]

        _hand_pos = _hand_pos + self.color_translate

        return _hand_pos

    def _getimg(self, _idx):
        for [_count_frames, _seq_name, _hand_poses] in self.seq_frames:
            if _idx >= _count_frames and _idx < (_count_frames + len(_hand_poses)):
                break
            continue
        _idx = _idx - _count_frames  

        _path = os.path.join(self.root_dir, _seq_name, "SK_color_%d.png" % (_idx))

        return cv2.imread(_path, cv2.IMREAD_COLOR)

focal_lengths = [607.92271, 607.88192]
center = [314.78337, 236.42484]

cam_intr = np.array([[focal_lengths[0], 0, center[0]],
                     [0, focal_lengths[1], center[1]],
                     [0, 0, 1]])

class Stb():

    def __init__(self, _train = False):
        #if Stb._globa_db == None:
        self._globa_db = StbBD()

        self.ids = _utils.db_ids(__file__, _train, len(self._globa_db))

    def __len__(self):
        return len(self.ids)

    def _uv2d(self, _skel3d):
        _skel_hom2d = cam_intr.dot(_skel3d.transpose()).transpose()
        _skel_2d = (_skel_hom2d / _skel_hom2d[:, 2:])[:, :2]
        return _skel_2d

    def __getitem__(self, _idx):
        _idx = self.ids[_idx]
        _skel3d = self._globa_db[_idx]

        return cam_intr, _skel3d, 'left'

    def getimg(self, _idx):
        _idx = self.ids[_idx]
        return self._globa_db._getimg(_idx)
