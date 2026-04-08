import numpy as np
from shapely.geometry import Polygon

def rotate_iou_cpu_eval(boxes, qboxes, criterion=-1):
    """
    CPU implementation of rotated IoU for KITTI evaluation.
    """
    overlaps = np.zeros((boxes.shape[0], qboxes.shape[0]), dtype=np.float32)

    for i in range(boxes.shape[0]):
        poly1 = _box_to_poly(boxes[i])
        for j in range(qboxes.shape[0]):
            poly2 = _box_to_poly(qboxes[j])

            inter = poly1.intersection(poly2).area
            union = poly1.union(poly2).area

            overlaps[i, j] = inter / union if union > 0 else 0.0

    return overlaps


def _box_to_poly(box):
    """
    Convert KITTI box format [x, y, w, l, heading] to a Shapely polygon.
    """
    x, y, w, l, heading = box
    c, s = np.cos(heading), np.sin(heading)

    dx, dy = w / 2, l / 2
    corners = np.array([
        [ dx,  dy],
        [-dx,  dy],
        [-dx, -dy],
        [ dx, -dy]
    ])

    rot = np.array([[c, -s], [s, c]])
    rotated = corners @ rot.T
    translated = rotated + np.array([x, y])

    return Polygon(translated)