import tensorflow as tf


# used for rpn prediction deltas with anchors, we can generate corresponding bbox
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

    # convert deltas to bbox
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
