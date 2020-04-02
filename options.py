from data.stb.stb import Stb

def pre_main_opts(parser):
    parser.add_argument(
        "--datasets",
        choices=[
            'Stb'
        ],
        nargs="+",

        default=[
            'Stb'],
        help="training dataset to load [Stb]",
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=112,
        help="input_size",
    )

    # prepocess
    parser.add_argument(
        "--pixel_mean",
        type=tuple,
        default=(127, 127, 127),
        help="mean",
    )

    parser.add_argument(
        "--pixel_std",
        type=tuple,
        default=(128, 128, 128),
        help="std",
    )
    # other
    parser.add_argument(
        "--num_thread",
        type=int,
        default=2,
        help="num_thread",
    )

    ## 
    parser.add_argument(
        "--model_dir",
        type=str,
        default=osp.join(osp.dirname(osp.abspath(__file__)), 'models'),
        help="2d head map or 3d position or 2d head map width 3d",
    )
    # training config
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=[0],
        help="training config gpus",
    )

    parser.add_argument(
        "--net_type",
        type=str,
        default='posesv2',
        help="training config continue_train",
    )

import os
import os.path as osp

def post_main_opts(config):
    _dbs = []
    _sides = []
    for _dataset in config.datasets:
        if _dataset == 'Stb':
            _dbs.append(Stb)
        else:
            raise ('don\' know db')

    config.db_classes = _dbs

    _gpus = ','.join([str(x) for x in config.gpus])
    os.environ["CUDA_VISIBLE_DEVICES"] = _gpus

    from data.data_pose import Pose2DDb
    config.dataset_class = Pose2DDb

    config.test_epoch = 125

import argparse

parser = argparse.ArgumentParser(description='graph')
pre_main_opts(parser)
args = parser.parse_args()
post_main_opts(args)
