import cv2
import numpy as np
"""
© Microsoft, 2017. Licensed under an MIT license.
Largely adataped from
https://github.com/JimmySuen/integral-human-pose
"""

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def transform_from_bbox(c_x, c_y, src_width, src_height, scale, rot, trans_x, trans_y, inv=False):
    """
    Create transformation with regard to a bounding box.
    args:
        c_x: center of 
    """
    #src width is the box
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    trans_w = src_width * trans_x
    trans_h = src_height * trans_y
    src_center = np.array([c_x + trans_w, c_y + trans_h], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    # translation


    # by changing dst width, we can also do resizing, instead do it in separate step?
    dst_w = src_width
    dst_h = src_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    # 3 src points
    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    # 3 dst points
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def transform_and_crop_image(cvimg, trans, bbox):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    # this does also the cropping
    bb_width, bb_height = bbox
    # the transformation moves the image so that
    # by setting the dst size, only the relevant parts are outputted
    # => cropping
    img_patch = cv2.warpAffine(img, trans, (int(bb_width), int(bb_height)), flags=cv2.INTER_LINEAR)
    return img_patch

def trans_point2d(pt_2d, trans):
    # add 3rd dim
    pt_3d = np.ones((pt_2d.shape[0], 3))
    pt_3d[:, :-1] = pt_2d
    dst_pt = np.dot(pt_3d[:, None, :], trans[None, :, :].transpose([0, 2, 1]))
    return dst_pt[:, 0:2].squeeze()
