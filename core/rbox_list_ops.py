# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Rotated Bounding Box List operations.
This code referred to 'box_list_ops.py'

Example rbox operations that are supported:
  * areas: compute bounding box areas
  * iou: pairwise intersection-over-union scores

Whenever rbox_list_ops functions output a RBoxList, the fields of the incoming
RBoxList are retained unless documented otherwise.
"""
from math import pi
import numpy as np
import tensorflow as tf

from core import rbox_list
from utils import shape_utils
import utils.np_rbox_ops as np_rbox_ops
from core import box_list_ops

import pyximport;

pyximport.install()
from utils import cintersection_rbox


class SortOrder(object):
    """Enum class for sort order.

    Attributes:
      ascend: ascend order.
      descend: descend order.
    """
    ascend = 1
    descend = 2


def area(rboxlist, scope=None):
    """Computes area of rboxes.

    Args:
      rboxlist: RBoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing box areas.
    """
    with tf.name_scope(scope, 'AreaRbox'):
        cy, cx, h, w, ang = tf.split(value=rboxlist.get(), num_or_size_splits=5, axis=1)
        return tf.squeeze(h * w, [1])


def height_width(rboxlist, scope=None):
    """Computes height and width of rboxes in rboxlist.

    Args:
      rboxlist: RBoxList holding N boxes
      scope: name scope.

    Returns:
      Height: A tensor with shape [N] representing box heights.
      Width: A tensor with shape [N] representing box widths.
    """
    with tf.name_scope(scope, 'HeightWidthRbox'):
        cy, cx, h, w, ang = tf.split(value=rboxlist.get(), num_or_size_splits=5, axis=1)
        return tf.squeeze(h, [1]), tf.squeeze(w, [1])


def scale(rboxlist, y_scale, x_scale, scope=None):
    """scale box coordinates in x and y dimensions.

    Args:
      rboxlist: RBoxList holding N boxes
      y_scale: (float) scalar tensor
      x_scale: (float) scalar tensor
      scope: name scope.

    Returns:
      rboxlist: RBoxList holding N boxes
    """
    with tf.name_scope(scope, 'ScaleRbox'):
        y_scale = tf.cast(y_scale, tf.float32)
        x_scale = tf.cast(x_scale, tf.float32)
        cy, cx, h, w, ang = tf.split(value=rboxlist.get(), num_or_size_splits=5, axis=1)
        cy = y_scale * cy
        cx = x_scale * cx
        h = y_scale * h
        w = x_scale * w
        scaled_rboxlist = rbox_list.RBoxList(tf.concat([cy, cx, h, w, ang], 1))
        return _copy_extra_fields(scaled_rboxlist, rboxlist)


def clip_to_window(rboxlist, window, filter_nonoverlapping=True, scope=None):
    pass


def prune_outside_window(rboxlist, window, scope=None):
    """Prunes bounding boxes that fall outside a given window.

      This function prunes rotated bounding boxes that even partially fall outside the given
      window. See also clip_to_window which only prunes bounding boxes that fall
      completely outside the window, and clips any bounding boxes that partially
      overflow.

      Args:
        rboxlist: a RBoxList holding M_in boxes.
        window: a float tensor of shape [4] representing [cy, cx, h, w, ang] of the window
        scope: name scope.

      Returns:
        pruned_corners: a tensor with shape [M_out, 5] where M_out <= M_in
        valid_indices: a tensor with shape [M_out] indexing the valid rotated bounding boxes in the input tensor.
      """
    with tf.name_scope(scope, 'PruneOutsideWindowRbox'):
        x, y = rboxlist.get_corners(separate_xy=True)
        win_y_min, win_x_min, win_y_max, win_x_max = tf.unstack(window)

        coordinate_violations = tf.concat([tf.less(y, win_y_min), tf.less(x, win_x_min),
                                           tf.greater(y, win_y_max), tf.greater(x, win_x_max)], 1)
        valid_indices = tf.reshape(tf.where(tf.logical_not(tf.reduce_any(coordinate_violations, 1))), [-1])
        return gather(rboxlist, valid_indices), valid_indices


def prune_completely_outside_window(boxlist, window, scope=None):
    pass


def intersection(rboxlist1, rboxlist2, ver='tf', ignore_large_than_pi_2=True, scope=None):
    """Compute pairwise intersection areas between rboxes.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ver: Implementation version which is one of 'shapely', 'opencv', 'tf' and 'cython'
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    if ver == 'shapely':
        return intersection_shapely(rboxlist1, rboxlist2, ignore_large_than_pi_2=ignore_large_than_pi_2, scope=scope)
    elif ver == 'opencv':
        return intersection_opencv(rboxlist1, rboxlist2, ignore_large_than_pi_2=ignore_large_than_pi_2, scope=scope)
    elif ver == 'tf':
        max_gts = 64
        n_box = rboxlist1.num_boxes()
        # if the number of rboxlist1 is larger than max_gts and ignore_large_than_pi_2 is true then apply map function.
        # ignore_large_than_pi_2 is true only when rboxlist1 is ground truths, i.e. training.
        do_map_fn = tf.logical_and(ignore_large_than_pi_2, tf.greater(n_box, max_gts))
        return tf.cond(do_map_fn,
                       lambda: intersection_tf_map(rboxlist1, rboxlist2, ignore_large_than_pi_2=ignore_large_than_pi_2,
                                                   scope=scope),
                       lambda: intersection_tf(rboxlist1, rboxlist2, ignore_large_than_pi_2=ignore_large_than_pi_2,
                                               scope=scope))
    else:  # cython
        return intersection_cython(rboxlist1, rboxlist2, ignore_large_than_pi_2=ignore_large_than_pi_2, scope=scope)


def intersection_shapely(rboxlist1, rboxlist2, ignore_large_than_pi_2=True, scope=None):
    """Compute pairwise intersection areas between rboxes by shapely.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'IntersectionRboxShapely'):
        corners1 = rboxlist1.get_corners()
        corners2 = rboxlist2.get_corners()

        [cy1, cx1, h1, w1, ang1] = tf.split(rboxlist1.get(), 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = tf.split(rboxlist2.get(), 5, axis=1)

        size = tf.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + tf.transpose(tf.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = tf.sqrt((cx1 - tf.transpose(cx2)) ** 2 + (cy1 - tf.transpose(cy2)) ** 2)
        if ignore_large_than_pi_2:
            is_diff_ang_large_then_pi_2 = tf.abs(ang1 - tf.transpose(ang2)) > pi / 2
        else:
            is_diff_ang_large_then_pi_2 = tf.zeros_like(dist, dtype=tf.bool)

        return tf.py_func(np_rbox_ops.intersection_shapely, [corners1, corners2, size, dist,
                                                             is_diff_ang_large_then_pi_2], tf.float32)


def intersection_opencv(rboxlist1, rboxlist2, ignore_large_than_pi_2=True, scope=None):
    """Compute pairwise intersection areas between rboxes by opencv.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'IntersectionRboxOpenCV'):
        [cy1, cx1, h1, w1, ang1] = tf.split(rboxlist1.get(), 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = tf.split(rboxlist2.get(), 5, axis=1)

        size = tf.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + tf.transpose(tf.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = tf.sqrt((cx1 - tf.transpose(cx2)) ** 2 + (cy1 - tf.transpose(cy2)) ** 2)
        if ignore_large_than_pi_2:
            is_diff_ang_large_then_pi_2 = tf.abs(ang1 - tf.transpose(ang2)) > pi / 2
        else:
            is_diff_ang_large_then_pi_2 = tf.zeros_like(dist, dtype=tf.bool)

        return tf.py_func(np_rbox_ops.intersection_opencv, [rboxlist1.get(), rboxlist2.get(), size, dist,
                                                            is_diff_ang_large_then_pi_2], tf.float32)


def intersection_cython(rboxlist1, rboxlist2, ignore_large_than_pi_2=True, scope=None):
    """Compute pairwise intersection areas between rboxes by cython.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'IntersectionRboxCython'):
        [cy1, cx1, h1, w1, ang1] = tf.split(rboxlist1.get(), 5, axis=1)
        [cy2, cx2, h2, w2, ang2] = tf.split(rboxlist2.get(), 5, axis=1)

        size = tf.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + tf.transpose(tf.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
        dist = tf.sqrt((cx1 - tf.transpose(cx2)) ** 2 + (cy1 - tf.transpose(cy2)) ** 2)
        if ignore_large_than_pi_2:
            is_diff_ang_large_then_pi_2 = tf.abs(ang1 - tf.transpose(ang2)) > pi / 2
            is_diff_ang_large_then_pi_2 = tf.cast(is_diff_ang_large_then_pi_2, tf.uint8)
        else:
            is_diff_ang_large_then_pi_2 = tf.zeros_like(dist, dtype=tf.uint8)

        return tf.py_func(cintersection_rbox.intersection_rbox, [rboxlist1.get(), rboxlist2.get(), size, dist,
                                                                 is_diff_ang_large_then_pi_2], tf.float32)


def intersection_tf(rboxlist1, rboxlist2, ignore_large_than_pi_2=True, scope=None):
    """Compute pairwise intersection areas between rboxes by tensorflow.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """
    with tf.name_scope(scope, 'IntersectionRboxTF'):
        def _intersection(rboxlist1, rboxlist2):
            # init
            boxes1 = rboxlist1.get_corners()
            boxes2 = rboxlist2.get_corners()

            x1 = boxes1[:, ::2]
            y1 = boxes1[:, 1::2]
            x2 = boxes2[:, ::2]
            y2 = boxes2[:, 1::2]

            # a line from p1 to p2: p1 + (p2-p1)*t, t=[0,1]
            _boxes1 = tf.concat([boxes1[:, 2:], boxes1[:, :2]], axis=1)
            _boxes2 = tf.concat([boxes2[:, 2:], boxes2[:, :2]], axis=1)
            vec1 = _boxes1 - boxes1
            vec2 = _boxes2 - boxes2
            vx1 = vec1[:, ::2]
            vy1 = vec1[:, 1::2]
            vx2 = vec2[:, ::2]
            vy2 = vec2[:, 1::2]

            # line test
            _x1 = shape_utils.repeat(x1, [1, 4])[:, tf.newaxis]
            _y1 = shape_utils.repeat(y1, [1, 4])[:, tf.newaxis]
            _x2 = tf.tile(x2, [1, 4])
            _y2 = tf.tile(y2, [1, 4])
            x21 = _x2 - _x1
            y21 = _y2 - _y1

            _vx1 = shape_utils.repeat(vx1, [1, 4])[:, tf.newaxis]
            _vy1 = shape_utils.repeat(vy1, [1, 4])[:, tf.newaxis]
            _vx2 = tf.tile(vx2, [1, 4])
            _vy2 = tf.tile(vy2, [1, 4])
            det = _vx2 * _vy1 - _vx1 * _vy2

            t1 = (_vx2 * y21 - _vy2 * x21) / det
            t2 = (_vx1 * y21 - _vy1 * x21) / det

            xi1 = tf.where((t1 >= 0.0) & (t1 <= 1.0) & (t2 >= 0.0) & (t2 <= 1.0), _x1 + _vx1 * t1,
                           tf.ones_like(t1) * (-np.inf))
            yi1 = tf.where((t1 >= 0.0) & (t1 <= 1.0) & (t2 >= 0.0) & (t2 <= 1.0), _y1 + _vy1 * t1,
                           tf.ones_like(t1) * (-np.inf))

            # Check for vertices from rect1 inside recct2
            x = _x1
            y = _y1
            A = -_vy2
            B = _vx2
            C = -(A * _x2 + B * _y2)
            s = A * x + B * y + C
            s1 = tf.where(s >= 0, tf.ones_like(s), s)
            s1 = tf.where(s1 <= 0, tf.ones_like(s) * -1, s1)

            s1_shape = tf.shape(s1)
            shape2 = tf.where(tf.equal(s1_shape[1], 0) | tf.equal(s1_shape[0], 0), 0, -1)
            s2 = tf.reshape(s1, [s1_shape[0], s1_shape[1], shape2, 4])
            s3 = tf.reduce_sum(s2, axis=3)
            __x1 = tf.tile(x1, [1, s1_shape[1]])
            __x1 = tf.reshape(__x1, tf.shape(s3))
            __y1 = tf.tile(y1, [1, s1_shape[1]])
            __y1 = tf.reshape(__y1, tf.shape(s3))
            xi2 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __x1, tf.ones_like(s3) * -np.inf)
            yi2 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __y1, tf.ones_like(s3) * -np.inf)

            # Reverse the check - check for vertices from rect2 inside recct1
            x = _x2
            y = _y2
            A = -_vy1
            B = _vx1
            C = -(A * _x1 + B * _y1)
            s = A * x + B * y + C
            s1 = tf.where(s >= 0, tf.ones_like(s), s)
            s1 = tf.where(s1 <= 0, tf.ones_like(s) * -1, s1)

            s1_shape = tf.shape(s1)
            shape2 = tf.where(tf.equal(s1_shape[1], 0) | tf.equal(s1_shape[0], 0), 0, -1)
            s2 = tf.reshape(s1, [s1_shape[0], s1_shape[1], shape2, 4])
            s3 = tf.reduce_sum(s2, axis=2)

            __x2 = tf.tile(x2, [s1_shape[0], 1])
            __x2 = tf.reshape(__x2, tf.shape(s3))
            __y2 = tf.tile(y2, [s1_shape[0], 1])
            __y2 = tf.reshape(__y2, tf.shape(s3))

            xi3 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __x2, tf.ones_like(s3) * -np.inf)
            yi3 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __y2, tf.ones_like(s3) * -np.inf)

            # concatenate intersection points
            xi = tf.concat([xi1, xi2, xi3], axis=2)
            yi = tf.concat([yi1, yi2, yi3], axis=2)

            ## acr sort
            n = tf.reduce_sum(tf.where(xi > -np.inf, tf.ones_like(xi), tf.zeros_like(xi)), axis=2)
            cx = tf.reduce_sum(tf.where(xi > -np.inf, xi, tf.zeros_like(xi)), axis=2) / n
            cy = tf.reduce_sum(tf.where(yi > -np.inf, yi, tf.zeros_like(xi)), axis=2) / n

            angles = tf.atan2(yi - cy[:, :, tf.newaxis], xi - cx[:, :, tf.newaxis])
            angles = tf.where(xi > -np.inf, angles, tf.ones_like(xi) * (-np.inf))

            _, indices = tf.nn.top_k(angles, tf.shape(angles)[2])

            xi_shape = tf.shape(xi)
            auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(xi_shape[:(xi.get_shape().ndims - 1)])
                                                                    + [tf.shape(angles)[2]])], indexing='ij')
            xi = tf.gather_nd(xi, tf.stack(auxiliary_indices[:-1] + [indices], axis=-1))
            yi = tf.gather_nd(yi, tf.stack(auxiliary_indices[:-1] + [indices], axis=-1))

            _n = tf.expand_dims(tf.cast(n - 1, tf.int32), -1)
            auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(xi_shape[:(xi.get_shape().ndims - 1)])
                                                                    + [1])], indexing='ij')
            xi_last = tf.gather_nd(xi, tf.stack(auxiliary_indices[:-1] + [_n], axis=-1))
            yi_last = tf.gather_nd(yi, tf.stack(auxiliary_indices[:-1] + [_n], axis=-1))

            xi = tf.concat([xi_last, xi], axis=-1)
            yi = tf.concat([yi_last, yi], axis=-1)

            # Polygon area
            x = tf.where(xi > -np.inf, xi, tf.zeros_like(xi))
            y = tf.where(yi > -np.inf, yi, tf.zeros_like(xi))

            x1y2 = x[:, :, :-1] * y[:, :, 1:]
            x2y1 = x[:, :, 1:] * y[:, :, :-1]

            area = -tf.reduce_sum(x1y2 - x2y1, axis=-1) * 0.5

            if ignore_large_than_pi_2:
                # Ignore rboxes that differ by more than pi/2
                [_, _, _, _, ang1] = tf.split(rboxlist1.get(), 5, axis=1)
                [_, _, _, _, ang2] = tf.split(rboxlist2.get(), 5, axis=1)
                diff_ang = tf.abs(ang1 - tf.transpose(ang2))
                area = tf.where(diff_ang > (pi / 2), -1 * tf.ones_like(area), area)

            return area

        n_boxes1 = rboxlist1.num_boxes()
        n_boxes2 = rboxlist2.num_boxes()

        empty = tf.zeros((n_boxes1, n_boxes2), dtype=tf.float32)
        return tf.cond(tf.equal(n_boxes1, 0) | tf.equal(n_boxes2, 0),
                       lambda: empty,
                       lambda: _intersection(rboxlist1, rboxlist2))


def intersection_tf_map(rboxlist1, rboxlist2, ignore_large_than_pi_2=True, scope=None):
    """Compute pairwise intersection areas between rboxes by tensorflow map_fn.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise intersections
    """

    boxes2 = rboxlist2.get_corners()
    with tf.name_scope(scope, 'IntersectionRboxTF'):
        def _intersection(rbox):
            # init
            rbox = tf.expand_dims(rbox, axis=0)
            [cy, cx, h, w, rbox_ang] = tf.split(rbox, 5, axis=1)
            h = h / 2
            w = w / 2
            cos = tf.cos(rbox_ang)
            sin = tf.sin(rbox_ang)

            lt_x = cx - w * cos + h * sin
            lt_y = cy - w * sin - h * cos
            rt_x = cx + w * cos + h * sin
            rt_y = cy + w * sin - h * cos
            lb_x = cx - w * cos - h * sin
            lb_y = cy - w * sin + h * cos
            rb_x = cx + w * cos - h * sin
            rb_y = cy + w * sin + h * cos

            boxes1 = tf.squeeze(tf.stack([lt_y, lt_x, rt_y, rt_x, rb_y, rb_x, lb_y, lb_x], 1), [2])
            # boxes2 = rboxlist2.get_corners()

            x1 = boxes1[:, ::2]
            y1 = boxes1[:, 1::2]
            x2 = boxes2[:, ::2]
            y2 = boxes2[:, 1::2]

            # a line from p1 to p2: p1 + (p2-p1)*t, t=[0,1]
            _boxes1 = tf.concat([boxes1[:, 2:], boxes1[:, :2]], axis=1)
            _boxes2 = tf.concat([boxes2[:, 2:], boxes2[:, :2]], axis=1)
            vec1 = _boxes1 - boxes1
            vec2 = _boxes2 - boxes2
            vx1 = vec1[:, ::2]
            vy1 = vec1[:, 1::2]
            vx2 = vec2[:, ::2]
            vy2 = vec2[:, 1::2]

            # line test
            _x1 = shape_utils.repeat(x1, [1, 4])[:, tf.newaxis]
            _y1 = shape_utils.repeat(y1, [1, 4])[:, tf.newaxis]
            _x2 = tf.tile(x2, [1, 4])
            _y2 = tf.tile(y2, [1, 4])
            x21 = _x2 - _x1
            y21 = _y2 - _y1

            _vx1 = shape_utils.repeat(vx1, [1, 4])[:, tf.newaxis]
            _vy1 = shape_utils.repeat(vy1, [1, 4])[:, tf.newaxis]
            _vx2 = tf.tile(vx2, [1, 4])
            _vy2 = tf.tile(vy2, [1, 4])
            det = _vx2 * _vy1 - _vx1 * _vy2

            t1 = (_vx2 * y21 - _vy2 * x21) / det
            t2 = (_vx1 * y21 - _vy1 * x21) / det

            xi1 = tf.where((t1 >= 0.0) & (t1 <= 1.0) & (t2 >= 0.0) & (t2 <= 1.0), _x1 + _vx1 * t1,
                           tf.ones_like(t1) * (-np.inf))
            yi1 = tf.where((t1 >= 0.0) & (t1 <= 1.0) & (t2 >= 0.0) & (t2 <= 1.0), _y1 + _vy1 * t1,
                           tf.ones_like(t1) * (-np.inf))

            # Check for vertices from rect1 inside recct2
            x = _x1
            y = _y1
            A = -_vy2
            B = _vx2
            C = -(A * _x2 + B * _y2)
            s = A * x + B * y + C
            s1 = tf.where(s >= 0, tf.ones_like(s), s)
            s1 = tf.where(s1 <= 0, tf.ones_like(s) * -1, s1)

            s1_shape = tf.shape(s1)
            shape2 = tf.where(tf.equal(s1_shape[1], 0) | tf.equal(s1_shape[0], 0), 0, -1)
            s2 = tf.reshape(s1, [s1_shape[0], s1_shape[1], shape2, 4])
            s3 = tf.reduce_sum(s2, axis=3)
            __x1 = tf.tile(x1, [1, s1_shape[1]])
            __x1 = tf.reshape(__x1, tf.shape(s3))
            __y1 = tf.tile(y1, [1, s1_shape[1]])
            __y1 = tf.reshape(__y1, tf.shape(s3))
            xi2 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __x1, tf.ones_like(s3) * -np.inf)
            yi2 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __y1, tf.ones_like(s3) * -np.inf)

            # Reverse the check - check for vertices from rect2 inside recct1
            x = _x2
            y = _y2
            A = -_vy1
            B = _vx1
            C = -(A * _x1 + B * _y1)
            s = A * x + B * y + C
            s1 = tf.where(s >= 0, tf.ones_like(s), s)
            s1 = tf.where(s1 <= 0, tf.ones_like(s) * -1, s1)

            s1_shape = tf.shape(s1)
            shape2 = tf.where(tf.equal(s1_shape[1], 0) | tf.equal(s1_shape[0], 0), 0, -1)
            s2 = tf.reshape(s1, [s1_shape[0], s1_shape[1], shape2, 4])
            s3 = tf.reduce_sum(s2, axis=2)

            __x2 = tf.tile(x2, [s1_shape[0], 1])
            __x2 = tf.reshape(__x2, tf.shape(s3))
            __y2 = tf.tile(y2, [s1_shape[0], 1])
            __y2 = tf.reshape(__y2, tf.shape(s3))

            xi3 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __x2, tf.ones_like(s3) * -np.inf)
            yi3 = tf.where((tf.equal(s3, 4) | tf.equal(s3, -4)), __y2, tf.ones_like(s3) * -np.inf)

            # concatenate intersection points
            xi = tf.concat([xi1, xi2, xi3], axis=2)
            yi = tf.concat([yi1, yi2, yi3], axis=2)

            ## acr sort
            n = tf.reduce_sum(tf.where(xi > -np.inf, tf.ones_like(xi), tf.zeros_like(xi)), axis=2)
            cx = tf.reduce_sum(tf.where(xi > -np.inf, xi, tf.zeros_like(xi)), axis=2) / n
            cy = tf.reduce_sum(tf.where(yi > -np.inf, yi, tf.zeros_like(xi)), axis=2) / n

            angles = tf.atan2(yi - cy[:, :, tf.newaxis], xi - cx[:, :, tf.newaxis])
            angles = tf.where(xi > -np.inf, angles, tf.ones_like(xi) * (-np.inf))

            _, indices = tf.nn.top_k(angles, tf.shape(angles)[2])

            xi_shape = tf.shape(xi)
            auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(xi_shape[:(xi.get_shape().ndims - 1)])
                                                                    + [tf.shape(angles)[2]])], indexing='ij')
            xi = tf.gather_nd(xi, tf.stack(auxiliary_indices[:-1] + [indices], axis=-1))
            yi = tf.gather_nd(yi, tf.stack(auxiliary_indices[:-1] + [indices], axis=-1))

            _n = tf.expand_dims(tf.cast(n - 1, tf.int32), -1)
            auxiliary_indices = tf.meshgrid(*[tf.range(d) for d in (tf.unstack(xi_shape[:(xi.get_shape().ndims - 1)])
                                                                    + [1])], indexing='ij')
            xi_last = tf.gather_nd(xi, tf.stack(auxiliary_indices[:-1] + [_n], axis=-1))
            yi_last = tf.gather_nd(yi, tf.stack(auxiliary_indices[:-1] + [_n], axis=-1))

            xi = tf.concat([xi_last, xi], axis=-1)
            yi = tf.concat([yi_last, yi], axis=-1)

            # Polygon area
            x = tf.where(xi > -np.inf, xi, tf.zeros_like(xi))
            y = tf.where(yi > -np.inf, yi, tf.zeros_like(xi))

            x1y2 = x[:, :, :-1] * y[:, :, 1:]
            x2y1 = x[:, :, 1:] * y[:, :, :-1]

            area = -tf.reduce_sum(x1y2 - x2y1, axis=-1) * 0.5

            if ignore_large_than_pi_2:
                # Ignore rboxes that differ by more than pi/2
                [_, _, _, _, ang2] = tf.split(rboxlist2.get(), 5, axis=1)
                diff_ang = tf.abs(rbox_ang - tf.transpose(ang2))
                area = tf.where(diff_ang > (pi / 2), -1 * tf.ones_like(area), area)

            return tf.squeeze(area)

        n_boxes1 = rboxlist1.num_boxes()
        n_boxes2 = rboxlist2.num_boxes()

        empty = tf.zeros((n_boxes1, n_boxes2), dtype=tf.float32)
        return tf.cond(tf.equal(n_boxes1, 0) | tf.equal(n_boxes2, 0),
                       lambda: empty,
                       lambda: tf.map_fn(_intersection, elems=rboxlist1.get(), dtype=tf.float32,
                                         parallel_iterations=64))


def matched_intersection(rboxlist1, rboxlist2, scope=None):
    """Compute intersection areas between corresponding rboxes in two rboxlists.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing pairwise intersections
    """
    with tf.name_scope(scope, 'MatchedIntersectionRbox'):
        return tf.py_func(np_rbox_ops.matched_intersection, [rboxlist1.get(), rboxlist2.get()], tf.float32)


def iou(rboxlist1, rboxlist2, ver='tf', ignore_large_than_pi_2=True, scope=None):
    """Computes pairwise intersection-over-union between box collections.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      ver: Implementation version which is one of 'shapely', 'opencv', 'tf' and 'cython'
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi
      is_approx: Whether or not calculate approximately IoU
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'IOURbox'):
        intersections = intersection(rboxlist1, rboxlist2, ver=ver, ignore_large_than_pi_2=ignore_large_than_pi_2)
        intersections = tf.reshape(intersections, [rboxlist1.num_boxes(), rboxlist2.num_boxes()])
        areas1 = area(rboxlist1)
        areas2 = area(rboxlist2)
        unions = (tf.expand_dims(areas1, 1) + tf.expand_dims(areas2, 0) - intersections)
        equal = tf.equal(intersections, 0.0)
        intersect_over_unions = tf.where(equal, tf.zeros_like(intersections), tf.truediv(intersections, unions))
        if ignore_large_than_pi_2:
            intersect_over_unions = tf.where(tf.equal(intersections, -1.), intersections, intersect_over_unions)

        return intersect_over_unions


def matched_iou(rboxlist1, rboxlist2, scope=None):
    """Compute intersection-over-union between corresponding rboxes in rboxlists.

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding N boxes
      scope: name scope.

    Returns:
      a tensor with shape [N] representing pairwise iou scores.
    """
    with tf.name_scope(scope, 'MatchedIOURbox'):
        intersections = matched_intersection(rboxlist1, rboxlist2)
        areas1 = area(rboxlist1)
        areas2 = area(rboxlist2)
        unions = areas1 + areas2 - intersections
        return tf.where(tf.equal(intersections, 0.0), tf.zeros_like(intersections), tf.truediv(intersections, unions))


def ioa(rboxlist1, rboxlist2, scope=None):
    """Computes pairwise intersection-over-area between rbox collections.

    intersection-over-area (IOA) between two rboxes box1 and box2 is defined as
    their intersection area over box2's area. Note that ioa is not symmetric,
    that is, ioa(box1, box2) != ioa(box2, box1).

    Args:
      rboxlist1: RBoxList holding N boxes
      rboxlist2: RBoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise ioa scores.
    """
    with tf.name_scope(scope, 'IOA'):
        intersections = intersection(rboxlist1, rboxlist2)
        intersections = tf.reshape(intersections, [rboxlist1.num_boxes(), rboxlist2.num_boxes()])
        areas = tf.expand_dims(area(rboxlist2), 0)
        return tf.truediv(intersections, areas)


def prune_non_overlapping_boxes(rboxlist1, rboxlist2, min_overlap=0.0, scope=None):
    """Prunes the boxes in boxlist1 that overlap less than thresh with boxlist2.

    For each box in rboxlist1, we want its IOA to be more than min_overlap with
    at least one of the boxes in rboxlist2. If it does not, we remove it.

    Args:
      rboxlist1: RBoxList holding N boxes.
      rboxlist2: RBoxList holding M boxes.
      min_overlap: Minimum required overlap between boxes, to count them as overlapping.
      scope: name scope.

    Returns:
      new_rboxlist1: A pruned rboxlist with size [N', 5].
      keep_inds: A tensor with shape [N'] indexing kept rotated bounding boxes in the first input BoxList `rboxlist1`.
    """
    with tf.name_scope(scope, 'PruneNonOverlappingRboxes'):
        ioa_ = ioa(rboxlist2, rboxlist1)  # [M, N] tensor
        ioa_ = tf.reduce_max(ioa_, reduction_indices=[0])  # [N] tensor
        keep_bool = tf.greater_equal(ioa_, tf.constant(min_overlap))
        keep_inds = tf.squeeze(tf.where(keep_bool), squeeze_dims=[1])
        new_boxlist1 = gather(rboxlist1, keep_inds)
        return new_boxlist1, keep_inds


def prune_small_boxes(rboxlist, min_side, scope=None):
    """Prunes small boxes in the rboxlist which have a side smaller than min_side.

    Args:
      rboxlist: RBoxList holding N boxes.
      min_side: Minimum width AND height of box to survive pruning.
      scope: name scope.

    Returns:
      A pruned rboxlist.
    """
    with tf.name_scope(scope, 'PruneSmallRboxes'):
        height, width = height_width(rboxlist)
        is_valid = tf.logical_and(tf.greater_equal(width, min_side),
                                  tf.greater_equal(height, min_side))
        return gather(rboxlist, tf.reshape(tf.where(is_valid), [-1]))


def change_coordinate_frame(rboxlist, window, scope=None):
    """Change coordinate frame of the rboxlist to be relative to window's frame.

    Given a window of the form [ymin, xmin, ymax, xmax],
    changes bounding box coordinates from boxlist to be relative to this window
    (e.g., the min corner maps to (0,0) and the max corner maps to (1,1)).

    An example use case is data augmentation: where we are given groundtruth
    boxes (rboxlist) and would like to randomly crop the image to some
    window (window). In this case we need to change the coordinate frame of
    each groundtruth rbox to be relative to this new window.

    Args:
      rboxlist: A RBoxList object holding N boxes.
      window: A rank 1 tensor [4].
      scope: name scope.

    Returns:
      Returns a RBoxList object with N boxes.
    """
    with tf.name_scope(scope, 'ChangeCoordinateFrameRbox'):
        win_height = window[2] - window[0]
        win_width = window[3] - window[1]
        rboxlist_new = scale(rbox_list.RBoxList(rboxlist.get() - [window[0], window[1], 0, 0, 0]),
                             1.0 / win_height, 1.0 / win_width)
        rboxlist_new = _copy_extra_fields(rboxlist_new, rboxlist)
        return rboxlist_new


def sq_dist(boxlist1, boxlist2, scope=None):
    """Computes the pairwise squared distances between box corners.

    This op treats each box as if it were a point in a 4d Euclidean space and
    computes pairwise squared distances.

    Mathematically, we are given two matrices of box coordinates X and Y,
    where X(i,:) is the i'th row of X, containing the 4 numbers defining the
    corners of the i'th box in boxlist1. Similarly Y(j,:) corresponds to
    boxlist2.  We compute
    Z(i,j) = ||X(i,:) - Y(j,:)||^2
           = ||X(i,:)||^2 + ||Y(j,:)||^2 - 2 X(i,:)' * Y(j,:),

    Args:
      boxlist1: BoxList holding N boxes
      boxlist2: BoxList holding M boxes
      scope: name scope.

    Returns:
      a tensor with shape [N, M] representing pairwise distances
    """
    with tf.name_scope(scope, 'SqDistRbox'):
        raise NotImplementedError


def boolean_mask(rboxlist, indicator, fields=None, scope=None):
    """Select boxes from RBoxList according to indicator and return new RBoxList.

    `boolean_mask` returns the subset of rboxes that are marked as "True" by the
    indicator tensor. By default, `boolean_mask` returns rboxes corresponding to
    the input index list, as well as all additional fields stored in the rboxlist
    (indexing into the first dimension).  However one can optionally only draw
    from a subset of fields.

    Args:
      rboxlist: RBoxList holding N boxes
      indicator: a rank-1 boolean tensor
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the box coordinates.
      scope: name scope.

    Returns:
      subboxlist: a RBoxList corresponding to the subset of the input BoxList specified by indicator
    Raises:
      ValueError: if `indicator` is not a rank-1 boolean tensor.
    """
    with tf.name_scope(scope, 'BooleanMaskRbox'):
        if indicator.shape.ndims != 1:
            raise ValueError('indicator should have rank 1')
        if indicator.dtype != tf.bool:
            raise ValueError('indicator should be a boolean tensor')
        subboxlist = rbox_list.RBoxList(tf.boolean_mask(rboxlist.get(), indicator))
        if fields is None:
            fields = rboxlist.get_extra_fields()
        for field in fields:
            if not rboxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = tf.boolean_mask(rboxlist.get_field(field), indicator)
            subboxlist.add_field(field, subfieldlist)
        return subboxlist


def gather(rboxlist, indices, fields=None, scope=None):
    """Gather boxes from RBoxList according to indices and return new RBoxList.

    By default, `gather` returns rboxes corresponding to the input index list, as
    well as all additional fields stored in the rboxlist (indexing into the
    first dimension).  However one can optionally only gather from a
    subset of fields.

    Args:
      rboxlist: RBoxList holding N boxes
      indices: a rank-1 tensor of type int32 / int64
      fields: (optional) list of fields to also gather from.  If None (default),
        all fields are gathered from.  Pass an empty fields list to only gather
        the rbox coordinates.
      scope: name scope.

    Returns:
      subboxlist: a RBoxList corresponding to the subset of the input BoxList
      specified by indices
    Raises:
      ValueError: if specified field is not contained in rboxlist or if the indices are not of type int32
    """
    with tf.name_scope(scope, 'GatherRbox'):
        if len(indices.shape.as_list()) != 1:
            raise ValueError('indices should have rank 1')
        if indices.dtype != tf.int32 and indices.dtype != tf.int64:
            raise ValueError('indices should be an int32 / int64 tensor')
        subboxlist = rbox_list.RBoxList(tf.gather(rboxlist.get(), indices))
        if fields is None:
            fields = rboxlist.get_extra_fields()
        for field in fields:
            if not rboxlist.has_field(field):
                raise ValueError('boxlist must contain all specified fields')
            subfieldlist = tf.gather(rboxlist.get_field(field), indices)
            subboxlist.add_field(field, subfieldlist)
        return subboxlist


def concatenate(rboxlists, fields=None, scope=None):
    """Concatenate list of RBoxLists.

    This op concatenates a list of input RBoxLists into a larger BoxList.  It also
    handles concatenation of RBoxList fields as long as the field tensor shapes
    are equal except for the first dimension.

    Args:
      rboxlists: list of RBoxList objects
      fields: optional list of fields to also concatenate.  By default, all
        fields from the first BoxList in the list are included in the concatenation.
      scope: name scope.

    Returns:
      a RBoxList with number of boxes equal to sum([rboxlist.num_boxes() for rboxlist in RBoxList])
    Raises:
      ValueError: if rboxlists is invalid (i.e., is not a list, is empty, or
        contains non RBoxList objects), or if requested fields are not contained in all rboxlists
    """
    with tf.name_scope(scope, 'ConcatenateRbox'):
        if not isinstance(rboxlists, list):
            raise ValueError('rboxlists should be a list')
        if not rboxlists:
            raise ValueError('rboxlists should have nonzero length')
        for rboxlist in rboxlists:
            if not isinstance(rboxlist, rbox_list.RBoxList):
                raise ValueError('all elements of rboxlists should be BoxList objects')
        concatenated = rbox_list.RBoxList(tf.concat([boxlist.get() for boxlist in rboxlists], 0))
        if fields is None:
            fields = rboxlists[0].get_extra_fields()
        for field in fields:
            first_field_shape = rboxlists[0].get_field(field).get_shape().as_list()
            first_field_shape[0] = -1
            if None in first_field_shape:
                raise ValueError('field %s must have fully defined shape except for the 0th dimension.' % field)
            for rboxlist in rboxlists:
                if not rboxlist.has_field(field):
                    raise ValueError('rboxlist must contain all requested fields')
                field_shape = rboxlist.get_field(field).get_shape().as_list()
                field_shape[0] = -1
                if field_shape != first_field_shape:
                    raise ValueError('field %s must have same shape for all boxlists '
                                     'except for the 0th dimension.' % field)
            concatenated_field = tf.concat([boxlist.get_field(field) for boxlist in rboxlists], 0)
            concatenated.add_field(field, concatenated_field)
        return concatenated


def sort_by_field(rboxlist, field, order=SortOrder.descend, scope=None):
    """Sort rboxes and associated fields according to a scalar field.

    A common use case is reordering the rboxes according to descending scores.

    Args:
      rboxlist: RBoxList holding N boxes.
      field: A BoxList field for sorting and reordering the BoxList.
      order: (Optional) descend or ascend. Default is descend.
      scope: name scope.

    Returns:
      sorted_rboxlist: A sorted RBoxList with the field in the specified order.

    Raises:
      ValueError: if specified field does not exist
      ValueError: if the order is not either descend or ascend
    """
    with tf.name_scope(scope, 'SortByFieldRbox'):
        if order != SortOrder.descend and order != SortOrder.ascend:
            raise ValueError('Invalid sort order')

        field_to_sort = rboxlist.get_field(field)
        if len(field_to_sort.shape.as_list()) != 1:
            raise ValueError('Field should have rank 1')

        num_boxes = rboxlist.num_boxes()
        num_entries = tf.size(field_to_sort)
        length_assert = tf.Assert(tf.equal(num_boxes, num_entries),
                                  ['Incorrect field size: actual vs expected.', num_entries, num_boxes])

        with tf.control_dependencies([length_assert]):
            # TODO: Remove with tf.device when top_k operation runs correctly on GPU.
            with tf.device('/cpu:0'):
                _, sorted_indices = tf.nn.top_k(field_to_sort, num_boxes, sorted=True)

        if order == SortOrder.ascend:
            sorted_indices = tf.reverse_v2(sorted_indices, [0])

        return gather(rboxlist, sorted_indices)


def visualize_boxes_in_image(image, boxlist, normalized=False, scope=None):
    """Overlay bounding box list on image.

    Currently this visualization plots a 1 pixel thick red bounding box on top
    of the image.  Note that tf.image.draw_bounding_boxes essentially is
    1 indexed.

    Args:
      image: an image tensor with shape [height, width, 3]
      boxlist: a BoxList
      normalized: (boolean) specify whether corners are to be interpreted
        as absolute coordinates in image space or normalized with respect to the
        image size.
      scope: name scope.

    Returns:
      image_and_boxes: an image tensor with shape [height, width, 3]
    """
    with tf.name_scope(scope, 'VisualizeBoxesInImageRbox'):
        raise NotImplementedError


def filter_field_value_equals(rboxlist, field, value, scope=None):
    """Filter to keep only rboxes with field entries equal to the given value.

    Args:
      rboxlist: RBoxList holding N boxes.
      field: field name for filtering.
      value: scalar value.
      scope: name scope.

    Returns:
      a BoxList holding M boxes where M <= N

    Raises:
      ValueError: if rboxlist not a RBoxList object or if it does not have the specified field.
    """
    with tf.name_scope(scope, 'FilterFieldValueEqualsRbox'):
        if not isinstance(rboxlist, rbox_list.RBoxList):
            raise ValueError('boxlist must be a BoxList')
        if not rboxlist.has_field(field):
            raise ValueError('boxlist must contain the specified field')
        filter_field = rboxlist.get_field(field)
        gather_index = tf.reshape(tf.where(tf.equal(filter_field, value)), [-1])
        return gather(rboxlist, gather_index)


def filter_greater_than(rboxlist, thresh, scope=None):
    """Filter to keep only rboxes with score exceeding a given threshold.

    This op keeps the collection of rboxes whose corresponding scores are
    greater than the input threshold.

    TODO: Change function name to FilterScoresGreaterThan

    Args:
      rboxlist: RBoxList holding N boxes.  Must contain a 'scores' field representing detection scores.
      thresh: scalar threshold
      scope: name scope.

    Returns:
      a BoxList holding M boxes where M <= N

    Raises:
      ValueError: if boxlist not a BoxList object or if it does not
        have a scores field
    """
    with tf.name_scope(scope, 'FilterGreaterThanRbox'):
        if not isinstance(rboxlist, rbox_list.RBoxList):
            raise ValueError('rboxlist must be a BoxList')
        if not rboxlist.has_field('scores'):
            raise ValueError('input rboxlist must have \'scores\' field')
        scores = rboxlist.get_field('scores')
        if len(scores.shape.as_list()) > 2:
            raise ValueError('Scores should have rank 1 or 2')
        if len(scores.shape.as_list()) == 2 and scores.shape.as_list()[1] != 1:
            raise ValueError('Scores should have rank 1 or have shape consistent with [None, 1]')
        high_score_indices = tf.cast(tf.reshape(tf.where(tf.greater(scores, thresh)), [-1]), tf.int32)
        return gather(rboxlist, high_score_indices)


def non_max_suppression(rboxlist, thresh, max_output_size, image_shape, intersection_tf=False, scope=None):
    """Non maximum suppression.

    This op greedily selects a subset of detection roated bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected rboxes.  Note that this only works for a single class ---
    to apply NMS to multi-class predictions, use MultiClassNonMaxSuppression.

    Args:
      rboxlist: RBoxList holding N boxes.  Must contain a 'scores' field representing detection scores.
      thresh: scalar threshold
      max_output_size: maximum number of retained boxes
      intersection_tf: (optional) Whether to use a tf version of the intersection.
      scope: name scope.

    Returns:
      a RBoxList holding M boxes where M <= max_output_size
    Raises:
      ValueError: if thresh is not in [0, 1]
    """
    with tf.name_scope(scope, 'NonMaxSuppressionRbox'):
        if not 0 <= thresh <= 1.0:
            raise ValueError('thresh must be between 0 and 1')
        if not isinstance(rboxlist, rbox_list.RBoxList):
            raise ValueError('rboxlist must be a BoxList')
        if not rboxlist.has_field('scores'):
            raise ValueError('input rboxlist must have \'scores\' field')

        absoluted_rboxlist = to_absolute_coordinates(rboxlist, image_shape[0], image_shape[1])

        if intersection_tf:
            sorted_rboxes = sort_by_field(absoluted_rboxlist, 'scores')
            indicator = tf.less(tf.range(sorted_rboxes.num_boxes()), max_output_size * 3)
            sorted_rboxes = boolean_mask(sorted_rboxes, indicator)
            intersect_over_union = iou(sorted_rboxes, sorted_rboxes, ignore_large_than_pi_2=False)
            indices = tf.py_func(np_rbox_ops.non_max_suppression, [sorted_rboxes.get(),
                                                                   sorted_rboxes.get_field('scores'),
                                                                   max_output_size,
                                                                   thresh,
                                                                   intersect_over_union], tf.int32)
            normalized_rboxlist = to_normalized_coordinates(sorted_rboxes, image_shape[0], image_shape[1])
        else:
            indices = tf.py_func(np_rbox_ops.non_max_suppression, [absoluted_rboxlist.get(),
                                                                   absoluted_rboxlist.get_field('scores'),
                                                                   max_output_size,
                                                                   thresh], tf.int32)

            normalized_rboxlist = to_normalized_coordinates(absoluted_rboxlist, image_shape[0], image_shape[1])

        indices = tf.reshape(indices, [-1])
        return gather(normalized_rboxlist, indices)


def _copy_extra_fields(rboxlist_to_copy_to, rboxlist_to_copy_from):
    """Copies the extra fields of rboxlist_to_copy_from to rboxlist_to_copy_to.

    Args:
      rboxlist_to_copy_to: RBoxList to which extra fields are copied.
      rboxlist_to_copy_from: RBoxList from which fields are copied.

    Returns:
      rboxlist_to_copy_to with extra fields.
    """
    for field in rboxlist_to_copy_from.get_extra_fields():
        rboxlist_to_copy_to.add_field(field, rboxlist_to_copy_from.get_field(field))
    return rboxlist_to_copy_to


def to_normalized_coordinates(rboxlist, height, width, check_range=False, scope=None):
    """Converts absolute box coordinates to normalized coordinates in [0, 1].

    Usually one uses the dynamic shape of the image or conv-layer tensor:
      rboxlist = rbox_list_ops.to_normalized_coordinates(rboxlist,
                                                         tf.shape(images)[1],
                                                         tf.shape(images)[2]),

    This function raises an assertion failed error at graph execution time when
    the maximum coordinate is smaller than 1.01 (which means that coordinates are
    already normalized). The value 1.01 is to deal with small rounding errors.

    Args:
      rboxlist: RBoxList with coordinates in terms of pixel-locations.
      height: Maximum value for height of absolute box coordinates.
      width: Maximum value for width of absolute box coordinates.
      check_range: If True, checks if the coordinates are normalized or not.
      scope: name scope.

    Returns:
      rboxlist with normalized coordinates in [0, 1].
    """
    with tf.name_scope(scope, 'ToNormalizedCoordinatesRbox'):
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)

        if check_range:
            max_val = tf.reduce_max(rboxlist.get())
            max_assert = tf.Assert(tf.greater(max_val, 1.01), ['max value is lower than 1.01: ', max_val])
            with tf.control_dependencies([max_assert]):
                width = tf.identity(width)

        return scale(rboxlist, 1 / height, 1 / width)


def to_absolute_coordinates(rboxlist, height, width, check_range=False, scope=None):
    """Converts normalized box coordinates to absolute pixel coordinates.

    This function raises an assertion failed error when the maximum box coordinate
    value is larger than 1.01 (in which case coordinates are already absolute).

    Args:
      rboxlist: BoxList with coordinates in range [0, 1].
      height: Maximum value for height of absolute box coordinates.
      width: Maximum value for width of absolute box coordinates.
      check_range: If True, checks if the coordinates are normalized or not.
      scope: name scope.

    Returns:
      rboxlist with absolute coordinates in terms of the image size.

    """
    with tf.name_scope(scope, 'ToAbsoluteCoordinatesRbox'):
        height = tf.cast(height, tf.float32)
        width = tf.cast(width, tf.float32)

        # Ensure range of input rboxes is correct.
        if check_range:
            cy, cx, _, _, _ = tf.split(value=rboxlist.get(), num_or_size_splits=5, axis=1)
            cycx = tf.stack([cy, cx])
            box_maximum = tf.reduce_max(cycx)
            max_assert = tf.Assert(tf.greater_equal(1.01, box_maximum),
                                   ['maximum box coordinate value is larger '
                                    'than 1.01: ', box_maximum])
            with tf.control_dependencies([max_assert]):
                width = tf.identity(width)

        return scale(rboxlist, height, width)


def refine_boxes_multi_class(pool_rboxes,
                             num_classes,
                             nms_iou_thresh,
                             nms_max_detections,
                             voting_iou_thresh=0.5):
    """Refines a pool of boxes using non max suppression and box voting.

    Box refinement is done independently for each class.

    Args:
      pool_rboxes: (BoxList) A collection of boxes to be refined. pool_boxes must
        have a rank 1 'scores' field and a rank 1 'classes' field.
      num_classes: (int scalar) Number of classes.
      nms_iou_thresh: (float scalar) iou threshold for non max suppression (NMS).
      nms_max_detections: (int scalar) maximum output size for NMS.
      voting_iou_thresh: (float scalar) iou threshold for box voting.

    Returns:
      BoxList of refined boxes.

    Raises:
      ValueError: if
        a) nms_iou_thresh or voting_iou_thresh is not in [0, 1].
        b) pool_boxes is not a BoxList.
        c) pool_boxes does not have a scores and classes field.
    """
    raise NotImplementedError


def refine_boxes(pool_rboxes,
                 nms_iou_thresh,
                 nms_max_detections,
                 voting_iou_thresh=0.5):
    """Refines a pool of boxes using non max suppression and box voting.

    Args:
      pool_rboxes: (BoxList) A collection of boxes to be refined. pool_boxes must
        have a rank 1 'scores' field.
      nms_iou_thresh: (float scalar) iou threshold for non max suppression (NMS).
      nms_max_detections: (int scalar) maximum output size for NMS.
      voting_iou_thresh: (float scalar) iou threshold for box voting.

    Returns:
      BoxList of refined boxes.

    Raises:
      ValueError: if
        a) nms_iou_thresh or voting_iou_thresh is not in [0, 1].
        b) pool_boxes is not a BoxList.
        c) pool_boxes does not have a scores field.
    """
    raise NotImplementedError


def box_voting(selected_boxes, pool_boxes, iou_thresh=0.5):
    """Performs box voting as described in S. Gidaris and N. Komodakis, ICCV 2015.

    Performs box voting as described in 'Object detection via a multi-region &
    semantic segmentation-aware CNN model', Gidaris and Komodakis, ICCV 2015. For
    each box 'B' in selected_boxes, we find the set 'S' of boxes in pool_boxes
    with iou overlap >= iou_thresh. The location of B is set to the weighted
    average location of boxes in S (scores are used for weighting). And the score
    of B is set to the average score of boxes in S.

    Args:
      selected_boxes: BoxList containing a subset of boxes in pool_boxes. These
        boxes are usually selected from pool_boxes using non max suppression.
      pool_boxes: BoxList containing a set of (possibly redundant) boxes.
      iou_thresh: (float scalar) iou threshold for matching boxes in
        selected_boxes and pool_boxes.

    Returns:
      BoxList containing averaged locations and scores for each box in
      selected_boxes.

    Raises:
      ValueError: if
        a) selected_boxes or pool_boxes is not a BoxList.
        b) if iou_thresh is not in [0, 1].
        c) pool_boxes does not have a scores field.
    """
    raise NotImplementedError


def pad_or_clip_box_list(rboxlist, num_boxes, scope=None):
    """Pads or clips all fields of a RBoxList.

    Args:
      rboxlist: A RBoxList with arbitrary of number of rboxes.
      num_boxes: First num_boxes in rboxlist are kept.
        The fields are zero-padded if num_boxes is bigger than the
        actual number of boxes.
      scope: name scope.

    Returns:
      RBoxList with all fields padded or clipped.
    """
    with tf.name_scope(scope, 'PadOrClipBoxListRbox'):
        subboxlist = rbox_list.RBoxList(shape_utils.pad_or_clip_tensor(rboxlist.get(), num_boxes))
        for field in rboxlist.get_extra_fields():
            subfield = shape_utils.pad_or_clip_tensor(rboxlist.get_field(field), num_boxes)
            subboxlist.add_field(field, subfield)
        return subboxlist


def convert_rboxes_to_boxes(rboxes, scope=None):
    """Convert rboxes to boxes

        Args:
          rboxes: boxes: a tensor of shape [N, 5] representing rbox.

        Returns:
          a tensor of shape [N, 4] representing box.
        """
    with tf.name_scope(scope, 'ConvertRbboxToBbox'):
        if isinstance(rboxes, rbox_list.RBoxList):
            rboxes = rboxes.get()

        # Get corner 4 points
        [cy, cx, h, w, ang] = tf.split(rboxes, 5, axis=1)
        h = h / 2
        w = w / 2
        cos = tf.cos(ang)
        sin = tf.sin(ang)

        lt_x = cx - w * cos + h * sin
        lt_y = cy - w * sin - h * cos
        rt_x = cx + w * cos + h * sin
        rt_y = cy + w * sin - h * cos
        lb_x = cx - w * cos - h * sin
        lb_y = cy - w * sin + h * cos
        rb_x = cx + w * cos - h * sin
        rb_y = cy + w * sin + h * cos

        # compute bounding boxes
        y_min = tf.minimum(tf.minimum(tf.minimum(lt_y, rt_y), lb_y), rb_y)
        x_min = tf.minimum(tf.minimum(tf.minimum(lt_x, rt_x), lb_x), rb_x)
        y_max = tf.maximum(tf.maximum(tf.maximum(lt_y, rt_y), lb_y), rb_y)
        x_max = tf.maximum(tf.maximum(tf.maximum(lt_x, rt_x), lb_x), rb_x)

        return tf.concat([y_min, x_min, y_max, x_max], 1)


def normalized_to_image_coordinates(normalized_boxes, image_shape, parallel_iterations=32):
    """Converts a batch of rboxes from normal to image coordinates.

    Args:
      normalized_boxes: a float32 tensor of shape [None, num_boxes, 5] in normalized coordinates.
      image_shape: a float32 tensor of shape [4] containing the image shape.
      parallel_iterations: parallelism for the map_fn op.

    Returns:
      absolute_boxes: a float32 tensor of shape [None, num_boxes, 5] containing the boxes in image coordinates.
    """

    def _to_absolute_coordinates(normalized_boxes):
        return to_absolute_coordinates(rbox_list.RBoxList(normalized_boxes),
                                       image_shape[1],
                                       image_shape[2],
                                       check_range=False).get()

    absolute_boxes = tf.map_fn(_to_absolute_coordinates,
                               elems=normalized_boxes,
                               dtype=tf.float32,
                               parallel_iterations=parallel_iterations,
                               back_prop=True)
    return absolute_boxes


def expand_rboxes(rboxes, ratio=2):
    """Expand rboxes for including context information.

    Args:
     rboxes: a float32 tensor of shape [None, num_boxes, 5] in normalized coordinates.
     ratio: Ratio for expanding area

    Returns:
     rboxes: Expanded rboxes
    """
    cy, cx, h, w, ang = tf.split(rboxes, num_or_size_splits=5, axis=1)
    w += (h * (ratio - 1))
    h *= ratio
    return tf.concat([cy, cx, h, w, ang], 1)
