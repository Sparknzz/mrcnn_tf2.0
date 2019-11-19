import tensorflow as tf


# NOTE used for trim the bboxes with 0 values. this maybe happen
# when do some argumentations which lead boxes out of the image
def trim_zeros(boxes, name=None):
    '''
    Often boxes are represented with matrices of shape [N, 4] and
    are padded with zeros. This removes zero boxes.

    Args
    ---
        boxes: [N, 4] matrix of boxes.
        non_zeros: [N] a 1D boolean mask identifying the rows to keep
    '''
    non_zeros = tf.cast(tf.reduce_sum(tf.abs(boxes), axis=1), dtype=tf.bool)
    boxes = tf.boolean_mask(boxes, non_zeros)

    return boxes, non_zeros


def compute_overlaps(anchors, gt_boxes):
    '''
        anchors: [N,4]
        gt_boxes: [N,4]

        return: [anchors, gts]
    '''
    # tile anchors and repeat gt_boxes
    b1 = tf.reshape(tf.tile(tf.expand_dims(anchors, 1), [1, 1, tf.shape(gt_boxes)[0]]), [-1, 4])
    b2 = tf.tile(gt_boxes, [tf.shape(anchors)[0], 1])

    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, -1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, -1)

    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)

    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)

    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)

    union = b1_area + b2_area - intersection

    iou = intersection / union  # eg anchor 20 gt 10, iou: [200,1]  reshape to [20, 10]

    overlaps = tf.reshape(iou, [tf.shape(anchors)[0], tf.shape(gt_boxes)[0]])

    return overlaps


def bbox2deltas(pos_anchor, gt, target_means=[0, 0, 0, 0], target_stds=[0.1, 0.1, 0.2, 0.2]):
    '''
    pos_anchor: [N,4]
    gt: [N,4]
    '''
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)

    pos_anchor = tf.cast(pos_anchor, tf.float32)
    gt = tf.cast(gt, tf.float32)

    height = pos_anchor[:, 2] - pos_anchor[:, 0]
    width = pos_anchor[:, 3] - pos_anchor[:, 1]
    center_y = pos_anchor[:, 0] + height / 2
    center_x = pos_anchor[:, 1] + width / 2

    gt_height = gt[:, 2] - gt[:, 0]
    gt_width = gt[:, 3] - gt[:, 1]
    gt_center_y = gt[:, 0] + gt_height / 2
    gt_center_x = gt[:, 1] + gt_width / 2

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    deltas = tf.stack([dy, dx, dh, dw], axis=-1)  # [N*1] => [N*4]

    deltas = (deltas - target_means) / target_stds

    return deltas


# used for rpn prediction deltas with anchors, we can generate corresponding rcnn
def deltas2bbox(anchors, deltas, target_means=[0, 0, 0, 0], target_stds=[0.1, 0.1, 0.2, 0.2]):
    '''Compute bounding box based on anchor and delta.
    Args
    ---
        anchors: [N, (y1, x1, y2, x2)] box to update
        deltas: [N, (dy, dx, log(dh), log(dw))] refinements to apply
        target_means: [4]
        target_stds: [4]
    '''
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)
    deltas = deltas * target_stds + target_means

    # convert to y,x,h,w
    height = anchors[:, 2] - anchors[:, 0]
    width = anchors[:, 3] - anchors[:, 1]
    center_y = anchors[:, 0] + 0.5 * height
    center_x = anchors[:, 1] + 0.5 * width

    # convert deltas to rcnn
    center_y += deltas[:, 0] * height
    center_x += deltas[:, 1] * width
    height *= tf.exp(deltas[:, 2])
    width *= tf.exp(deltas[:, 3])

    # convert to y1,x1,y2,x2
    y1 = center_y - 0.5 * height
    x1 = center_x - 0.5 * width
    y2 = y1 + height
    x2 = x1 + width

    result = tf.stack([y1, x1, y2, x2], axis=1)

    return result
