
from torch.utils.data import DataLoader
import numpy as np
import torch
from options import args

if __name__ == "__main__":

    _path = "{}/{}_{}.tar".format(args.model_dir, args.net_type, args.test_epoch)
    _module = torch.load(_path)
    if torch.cuda.is_available():
        _module = _module.cuda()
    ##
    _mean = np.array(args.pixel_mean, dtype = np.float32).reshape((1, -1))
    _std = np.array(args.pixel_std, dtype = np.float32).reshape((1, -1))

    _mean = torch.as_tensor(_mean, dtype=torch.float32)
    _std = torch.as_tensor(_std, dtype=torch.float32)

    def _normalize(_input_tensor):
        _input_tensor.sub_(_mean[:, :, None, None]).div_(_std[:, :, None, None])


    _dataset = args.dataset_class(args, True)
    _loader = DataLoader(dataset = _dataset, batch_size = 1, shuffle = True, num_workers= 0, pin_memory = True)

    for _itr, _data in enumerate(_loader):
        _img, _uv2d, _side = _data

        import copy
        _img_copy = copy.deepcopy(_img)

        _normalize(_img)
        if torch.cuda.is_available():
            _img = _img.cuda()

        _coord2d = _module(_img)

        import numpy as np
        from matplotlib import pyplot as plt
        from data.visual import _vis_point, _vis_edge, _vis_line

        _pose = _coord2d
        _pose = _pose.cpu().detach().numpy()

        _img = _img_copy

        _img = _img.numpy()
        _uv2d = _uv2d.numpy()

        _b, _c, _h, _w  = _img.shape
        _pose[:, :, 0] = _pose[:, :, 0] * _w
        _pose[:, :, 1] = _pose[:, :, 1] * _h

        _uv2d[:, :, 0] = _uv2d[:, :, 0] * _w
        _uv2d[:, :, 1] = _uv2d[:, :, 1] * _h

        for _id in range(_b):
            fig = plt.figure(figsize=(8, 8))
            ax1 = fig.add_subplot(221)
            ax1.set_title('img_pose')
            ax2 = fig.add_subplot(222)
            ax2.set_title('img_gt')
            ax3 = fig.add_subplot(223)
            ax3.set_title('img_pose_gt')
            ax4 = fig.add_subplot(224)
            ax4.set_title('pose_gt')

            _vis_point(_pose[_id], img = _img[_id].transpose((1,2,0)), ax = ax1)
            _vis_edge(_pose[_id], ax = ax1)

            _vis_point(_uv2d[_id], img = _img[_id].transpose((1,2,0)), ax = ax2)
            _vis_edge(_uv2d[_id], ax = ax2)

            _vis_point(_pose[_id], img = _img[_id].transpose((1,2,0)), ax = ax3)
            _vis_edge(_pose[_id], ax = ax3)
            _vis_point(_uv2d[_id], ax = ax3)
            _vis_edge(_uv2d[_id], ax = ax3)

            _vis_point(_pose[_id], ax = ax4)
            _vis_edge(_pose[_id], ax = ax4)
            _vis_point(_uv2d[_id], ax = ax4)
            _vis_edge(_uv2d[_id], ax = ax4)

            plt.show()
