import logging

logger = logging.getLogger(__name__)

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Decimal Code (R,G,B)
edge_color = [
    [255, 0, 0],
    [125, 125, 0],
    [0,  125, 255],
    [255, 125, 0],
    [0, 255, 125],
    [255, 0, 125],
]

point_color = [250, 150, 150]

def _vis_image(img, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    H, W, C = img.shape
    if C == 1:
        # remove channnel dimension
        ax.imshow(img.squeeze())
    else:
        ax.imshow(img.astype(np.uint8))

    if W is not None:
        ax.set_xlim(left=0, right=W)
    if H is not None:
        ax.set_ylim(bottom=H - 1, top=0)

    ax.scatter(W /2 , H /2 , c='r')
    return ax

def _vis_point(point, img=None, color=point_color, ax=None):
    if ax is None:
        fig = plt.figure()
        if point.shape[-1] == 3:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1)

    c = np.asarray(color) / 255. if color is not None else None

    pts = point.transpose()
    ax.scatter(*pts, c=c)

    if img is None:
        return ax

    _vis_image(img, ax)
    return ax

import data.utils as _utils

def _vis_edge(point, img=None, color=edge_color, ax=None):
    if ax is None:
        fig = plt.figure()
        if point.shape[-1] == 3:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1)

    if color is not None:
        color = np.asarray(color) / 255.
    else:
        color = [None] * len(_utils._links())

    for (_link, c) in zip(_utils._links(), color):
        _line = point[list(_link)]
        _line = _line.transpose()

        ax.plot(*_line, c=c)

    if img is None:
        return ax

    _vis_image(img, ax)
    return ax

def _vis_line(_start_point, _end_point, color=edge_color, ax=None):
    if ax is None:
        fig = plt.figure()
        if _start_point.shape[-1] == 3:
            ax = fig.add_subplot(1, 1, 1, projection="3d")
        else:
            ax = fig.add_subplot(1, 1, 1)

    if color is not None:
        color = np.asarray(color) / 255.
    else:
        color = [None] * len(_utils._links())

    for _id in range(len(_start_point)):
        _start = _start_point[_id]
        _end = _end_point[_id]

        _c = color[_id % len(color)]


        _line = np.array([_start, _end], dtype=np.float)
        _line = _line.transpose()

        ax.plot(*_line, c=_c)

    return ax
