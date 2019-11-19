import cv2
import numpy as np

#cython: boundscheck=False

def intersection_rbox(boxes1, boxes2, float[:,:] size, float[:,:] dist, unsigned char[:,:] is_diff_ang_large_then_pi_2):
    """Compute pairwise intersection areas between boxes.

    Args:
      boxes1: a numpy array with shape [N, 5] holding N boxes
      boxes2: a numpy array with shape [M, 5] holding M boxes

    Returns:
      a numpy array with shape [N*M] representing pairwise intersection area
    """
    [cy1, cx1, h1, w1, ang1] = np.split(boxes1, 5, axis=1)
    [cy2, cx2, h2, w2, ang2] = np.split(boxes2, 5, axis=1)

    cdef unsigned int idx1, idx2
    cdef unsigned int n_boxes1 = len(boxes1)
    cdef unsigned int n_boxes2 = len(boxes2)
    cdef float pi = 3.14159265358979323846264338327950288

    cx1 = np.squeeze(cx1, axis=1)
    cy1 = np.squeeze(cy1, axis=1)
    h1 = np.squeeze(h1, axis=1)
    w1 = np.squeeze(w1, axis=1)
    ang1 = np.squeeze(ang1, axis=1)
    cdef float[:] _cx1 = cx1
    cdef float[:] _cy1 = cy1
    cdef float[:] _h1 = h1
    cdef float[:] _w1 = w1
    cdef float[:] _ang1 = ang1

    cx2 = np.squeeze(cx2, axis=1)
    cy2 = np.squeeze(cy2, axis=1)
    h2 = np.squeeze(h2, axis=1)
    w2 = np.squeeze(w2, axis=1)
    ang2 = np.squeeze(ang2, axis=1)
    cdef float[:] _cx2 = cx2
    cdef float[:] _cy2 = cy2
    cdef float[:] _h2 = h2
    cdef float[:] _w2 = w2
    cdef float[:] _ang2 = ang2

    cdef float[:,:] intersections = np.empty((n_boxes1, n_boxes2), dtype=np.float32)

    for idx1 in range(n_boxes1):
        inter_areas = []
        for idx2 in range(n_boxes2):
            # ignore rbox large than difference angle |pi/2|
            if is_diff_ang_large_then_pi_2[idx1, idx2]:
                intersections[idx1, idx2] = -1
            elif dist[idx1, idx2] > size[idx1, idx2]:
                intersections[idx1, idx2] = 0
            else:
                try:
                    _, inter_points = cv2.rotatedRectangleIntersection(
                        ((_cx1[idx1], _cy1[idx1]), (_w1[idx1], _h1[idx1]), _ang1[idx1]*180/pi),
                        ((_cx2[idx2], _cy2[idx2]), (_w2[idx2], _h2[idx2]), _ang2[idx2]*180/pi))

                    if inter_points is not None:
                        center = inter_points.mean(axis=0)
                        angle = np.arctan2(inter_points[:, :, 1] - center[:, 1], inter_points[:, :, 0] - center[:, 0])
                        sort_indices = angle.argsort(axis=0).squeeze()
                        area = cv2.contourArea(inter_points[sort_indices])
                    else:
                        area = 0
                except Exception as e:
                    print(e)
                    area = (_w1[idx1] * _h1[idx1] + _w2[idx2] * _h2[idx2]) / 2

                intersections[idx1, idx2] = area

    return intersections
