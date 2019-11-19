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

"""Operations for [N, 5] numpy arrays representing rotated bounding boxes.

Example box operations that are supported:
  * Areas: compute bounding box areas
  * IOU: pairwise intersection-over-union scores
  * Non maximum Suppression
"""
from math import pi
import numpy as np
import cv2
from shapely.geometry import Polygon
from shapely.affinity import rotate

import pyximport; pyximport.install()
from utils import cintersection_rbox


class SortOrder(object):
    """Enum class for sort order.

    Attributes:
      ascend: ascend order.
      descend: descend order.
    """
    ASCEND = 1
    DESCEND = 2


def area(rboxes):
    """Computes area of rboxes.

    Args:
      rboxes: Numpy array with shape [N, 5] holding N rboxes

    Returns:
      a numpy array with shape [N*1] representing rbox areas
    """
    return rboxes[:, 2] * rboxes[:, 3]


def intersection_shapely_and_find_corner(rboxes1, rboxes2):
    """Compute pairwise intersection areas between rboxes.

    Notice that it find corners and compute intersection areas.

    Args:
      rboxes1: a numpy array with shape [N, 5] holding N rboxes
      rboxes2: a numpy array with shape [M, 5] holding M rboxes

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [cy1, cx1, h1, w1, ang1] = np.split(rboxes1, 5, axis=1)
    [cy2, cx2, h2, w2, ang2] = np.split(rboxes2, 5, axis=1)

    polygons1 = []
    polygons2 = []
    for cy, cx, h, w, ang in zip(cy1, cx1, h1, w1, ang1):
        polygon = Polygon([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2], [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]])
        polygons1.append(rotate(polygon, ang, use_radians=True))
    for cy, cx, h, w, ang in zip(cy2, cx2, h2, w2, ang2):
        polygon = Polygon([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2], [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]])
        polygons2.append(rotate(polygon, ang, use_radians=True))

    pi_2 = pi / 2
    intersections = []
    for polygon1, _ang1 in zip(polygons1, ang1):
        inter_areas = []
        for polygon2, _ang2 in zip(polygons2, ang2):
            # ignore rbox large than difference angle |pi/2|
            if abs(_ang1 - _ang2) >= pi_2:
                inter_areas.append(-1)
            else:
                inter = polygon1.intersection(polygon2)
                inter_areas.append(inter.area)

        intersections.append(inter_areas)
    return np.reshape(np.array(intersections, dtype=np.float32), (len(polygons1), len(polygons2)))


def intersection_shapely(rboxes1, rboxes2, size, dist, is_diff_ang_large_then_pi_2):
    """Compute pairwise intersection areas between rboxes by shapely.

    Args:
      rboxes1: a numpy array with shape [N, 8] holding N rboxes representing rbox corners
      rboxes2: a numpy array with shape [M, 8] holding M rboxes representing rbox corners
      size: a numpy array with shape [N, M] representing size of boxes1 + boxes2
      dist: a numpy array with shape [N, M] representing distance between boxes1 and boxes2
      is_diff_ang_large_then_pi_2: a numpy array with shape [N, M] whether or not a difference angle is large then pi/2

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [lt_y1, lt_x1, rt_y1, rt_x1, rb_y1, rb_x1, lb_y1, lb_x1] = np.split(rboxes1, 8, axis=1)
    [lt_y2, lt_x2, rt_y2, rt_x2, rb_y2, rb_x2, lb_y2, lb_x2] = np.split(rboxes2, 8, axis=1)

    polygons1 = []
    polygons2 = []
    for lt_y, lt_x, rt_y, rt_x, rb_y, rb_x, lb_y, lb_x\
            in zip(lt_y1, lt_x1, rt_y1, rt_x1, rb_y1, rb_x1, lb_y1, lb_x1):
        polygons1.append(Polygon([(lt_x, lt_y), (rt_x, rt_y), (rb_x, rb_y), (lb_x, lb_y)]))
    for lt_y, lt_x, rt_y, rt_x, rb_y, rb_x, lb_y, lb_x\
            in zip(lt_y2, lt_x2, rt_y2, rt_x2, rb_y2, rb_x2, lb_y2, lb_x2):
        polygons2.append(Polygon([(lt_x, lt_y), (rt_x, rt_y), (rb_x, rb_y), (lb_x, lb_y)]))

    intersections = []
    for idx1, polygon1 in enumerate(polygons1):
        inter_areas = []
        for idx2, polygon2 in enumerate(polygons2):
            # ignore rbox large than difference angle |pi/2|
            if is_diff_ang_large_then_pi_2[idx1, idx2]:
                inter_areas.append(-1)
            elif dist[idx1, idx2] > size[idx1, idx2]:
                inter_areas.append(0)
            else:
                inter = polygon1.intersection(polygon2)
                inter_areas.append(inter.area)
        intersections.append(inter_areas)

    return np.reshape(np.array(intersections, dtype=np.float32), (len(polygons1), len(polygons2)))


def intersection_opencv(rboxes1, rboxes2, size, dist, is_diff_ang_large_then_pi_2):
    """Compute pairwise intersection areas between rboxes by opencv.

    Args:
      rboxes1: a numpy array with shape [N, 8] holding N rboxes representing rbox corners
      rboxes2: a numpy array with shape [M, 8] holding M rboxes representing rbox corners
      size: a numpy array with shape [N, M] representing size of boxes1 + boxes2
      dist: a numpy array with shape [N, M] representing distance between boxes1 and boxes2
      is_diff_ang_large_then_pi_2: a numpy array with shape [N, M] whether or not a difference angle is large then pi/2

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [cy1, cx1, h1, w1, ang1] = np.split(rboxes1, 5, axis=1)
    [cy2, cx2, h2, w2, ang2] = np.split(rboxes2, 5, axis=1)

    intersections = []
    for idx1 in range(len(rboxes1)):
        inter_areas = []
        for idx2 in range(len(rboxes2)):
            # ignore rbox large than difference angle |pi/2|
            if is_diff_ang_large_then_pi_2[idx1, idx2]:
                inter_areas.append(-1)
            elif dist[idx1, idx2] > size[idx1, idx2]:
                inter_areas.append(0)
            else:

                _, inter_points = cv2.rotatedRectangleIntersection(
                    ((cx1[idx1], cy1[idx1]), (w1[idx1], h1[idx1]), ang1[idx1]*180/pi),
                    ((cx2[idx2], cy2[idx2]), (w2[idx2], h2[idx2]), ang2[idx2]*180/pi))

                if inter_points is not None:
                    center = inter_points.mean(axis=0)
                    angle = np.arctan2(inter_points[:, :, 1] - center[:, 1], inter_points[:, :, 0] - center[:, 0])
                    sort_indices = angle.argsort(axis=0).squeeze()
                    area = cv2.contourArea(inter_points[sort_indices])
                else:
                    area = 0
                inter_areas.append(area)
        intersections.append(inter_areas)

    return np.reshape(np.array(intersections, dtype=np.float32), (len(rboxes1), len(rboxes2)))


def matched_intersection(rboxes1, rboxes2):
    """Compute intersection areas between corresponding boxes in two rboxlists by shapely.

    Args:
      rboxes1: a numpy array with shape [N, 5] holding N rboxes
      rboxes2: a numpy array with shape [M, 5] holding M rboxes
    scope: name scope.

    Returns:
      a tensor with shape [N] representing pairwise intersections
    """
    [cy1, cx1, h1, w1, ang1] = np.split(rboxes1, 5, axis=1)
    [cy2, cx2, h2, w2, ang2] = np.split(rboxes2, 5, axis=1)

    polygons1 = []
    polygons2 = []
    for cy, cx, h, w, ang in zip(cy1, cx1, h1, w1, ang1):
        polygon = Polygon([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2], [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]])
        polygons1.append(rotate(polygon, ang, use_radians=True))
    for cy, cx, h, w, ang in zip(cy2, cx2, h2, w2, ang2):
        polygon = Polygon([[cx-w/2, cy-h/2], [cx+w/2, cy-h/2], [cx+w/2, cy+h/2], [cx-w/2, cy+h/2]])
        polygons2.append(rotate(polygon, ang, use_radians=True))

    intersections = []
    for polygon1, _ang1, polygon2, _ang2 in zip(polygons1, ang1, polygons2, ang2):
        if abs(_ang1 - _ang2) > pi / 2:
            intersections.append(-1)
        else:
            intersections.append(polygon1.intersection(polygon2).area)

    return np.array(intersections, dtype=np.float32)


def iou(rboxes1, rboxes2, ignore_large_than_pi_2=True):
    """Computes pairwise intersection-over-union between rbox collections.

    Notice that cython version of intersection is default.

    Args:
      rboxes1: a numpy array with shape [N, 5] holding N rboxes.
      rboxes2: a numpy array with shape [M, 5] holding N rboxes.
      ignore_large_than_pi_2: Whether or not ignore boxes with difference of angles larger than 2pi

    Returns:
      a numpy array with shape [N, M] representing pairwise iou scores.
    """
    [cy1, cx1, h1, w1, ang1] = np.split(rboxes1, 5, axis=1)
    [cy2, cx2, h2, w2, ang2] = np.split(rboxes2, 5, axis=1)

    size = np.sqrt((h1 / 2) ** 2 + (w1 / 2) ** 2) + np.transpose(np.sqrt((h2 / 2) ** 2 + (w2 / 2) ** 2))
    dist = np.sqrt((cx1 - np.transpose(cx2)) ** 2 + (cy1 - np.transpose(cy2)) ** 2)

    if ignore_large_than_pi_2:
        is_diff_ang_large_then_pi_2 = np.abs(ang1 - np.transpose(ang2)) > pi / 2
        is_diff_ang_large_then_pi_2 = is_diff_ang_large_then_pi_2.astype(np.uint8)
    else:
        is_diff_ang_large_then_pi_2 = np.zeros_like(dist, dtype=np.uint8)

    intersect = cintersection_rbox.intersection_rbox(rboxes1, rboxes2, size, dist, is_diff_ang_large_then_pi_2)

    area1 = area(rboxes1)
    area2 = area(rboxes2)
    union = np.expand_dims(area1, axis=1) + np.expand_dims(area2, axis=0) - intersect
    return intersect / union


def sort_by_score(rboxes, scores, order=SortOrder.DESCEND):
    """Sort rboxes by scores.

     Args:
       rboxes: a numpy array with shape [N, 5] holding N rboxes
       scores: A BoxList field for sorting and reordering the BoxList.
       order: (Optional) descend or ascend. Default is descend.

     Returns:
       sorted_rboxes: Sorted rboxes in the specified order.
       sorted_indices: Sorted indices in the specified order.

     Raises:
       ValueError: if the order is not either descend or ascend
     """
    if order != SortOrder.DESCEND and order != SortOrder.ASCEND:
        raise ValueError('Invalid sort order')

    sorted_indices = np.argsort(scores).astype(np.int32)
    if order == SortOrder.DESCEND:
        sorted_indices = sorted_indices[::-1]
    return rboxes[sorted_indices, :], sorted_indices


def non_max_suppression(rboxes,
                        scores,
                        max_output_size=10000,
                        iou_threshold=1.0,
                        intersect_over_union=None):
    """Non maximum suppression.

    This op greedily selects a subset of detection rotated bounding boxes, pruning
    away boxes that have high IOU (intersection over union) overlap (> thresh)
    with already selected boxes. In each iteration, the detected rotated bounding box with
    highest score in the available pool is selected.

    Args:
      rboxes: a numpy array with shape [N, 5] holding N rboxes.
      scores: a numpy array with shape [N] representing a single score corresponding to each rbox.
      max_output_size: maximum number of retained boxes
      iou_threshold: intersection over union threshold.
      intersect_over_union: (optional) if it isn't null, use values of intersect_over_union for iou.

    Returns:
      a numpy array holding indices of rboxes.

    Raises:
      ValueError: if threshold is not in [0, 1]
      ValueError: if max_output_size < 0
    """
    if rboxes.shape[0] != scores.shape[0]:
        raise ValueError('sizes of boxes and scores must be same.')
    if iou_threshold < 0. or iou_threshold > 1.0:
        raise ValueError('IOU threshold must be in [0, 1]')
    if max_output_size < 0:
        raise ValueError('max_output_size must be bigger than 0.')

    # Input rboxes had to be filtered.
    num_boxes = rboxes.shape[0]
    if num_boxes == 0:
        return np.array([], dtype=np.int32)

    if intersect_over_union is None:
        rboxes, sorted_indices = sort_by_score(rboxes, scores)
    else:
        sorted_indices = np.arange(num_boxes, dtype=np.int32)

    # Prevent further computation if NMS is disabled.
    if iou_threshold == 1.0:
        if num_boxes > max_output_size:
            return sorted_indices[:max_output_size]
        else:
            return sorted_indices[:num_boxes]

    # is_index_valid is True only for all remaining valid boxes,
    is_index_valid = np.full(num_boxes, 1, dtype=bool)
    selected_indices = []
    num_output = 0
    for i in range(num_boxes):
        if num_output < max_output_size:
            if is_index_valid[i]:
                num_output += 1
                selected_indices.append(i)
                is_index_valid[i] = False
                valid_indices = np.where(is_index_valid)[0]
                if valid_indices.size == 0:
                    break

                if intersect_over_union is not None:
                    is_index_valid[valid_indices] = \
                        np.logical_and(is_index_valid[valid_indices],
                                       intersect_over_union[i, valid_indices] <= iou_threshold)
                else:
                    _intersect_over_union = iou(np.expand_dims(rboxes[i, :], axis=0),
                                                rboxes[valid_indices, :], ignore_large_than_pi_2=False)
                    _intersect_over_union = np.squeeze(_intersect_over_union, axis=0)
                    is_index_valid[valid_indices] = np.logical_and(is_index_valid[valid_indices],
                                                                   _intersect_over_union <= iou_threshold)

    return np.array(sorted_indices[selected_indices], dtype=np.int32)
