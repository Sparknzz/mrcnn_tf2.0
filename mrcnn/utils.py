import tensorflow as tf


def compute_overlaps(anchors, gt_boxes):
    '''
    Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)]. normalized coordinates
    '''
    # 1. Tile gt_boxes and repeate anchors. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    # each anchor need to compare with all gt!! so repeat gt times for each anchor
    b1 = tf.reshape(tf.tile(tf.expand_dims(anchors, 1), [1, 1, tf.shape(gt_boxes)[0]]), [-1, 4])
    # repeat anchors times
    b2 = tf.tile(gt_boxes, [tf.shape(anchors)[0], 1])

    # 2. compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)

    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(y2 - y1, 0) * tf.maximum(x2 - x1, 0)

    # 3. compute union
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    unions = b1_area + b2_area - intersection

    iou = intersection / unions  # [b1 * b2, 1]

    overlaps = tf.reshape(iou, [tf.shape(anchors)[0], tf.shape(gt_boxes)[0]])
    return overlaps


def bbox2deltas(anchors, gt_boxes, target_means, target_stds):
    '''Compute refinement needed to transform box to gt_box.

    Args
    ---
        box: [..., (y1, x1, y2, x2)]
        gt_box: [..., (y1, x1, y2, x2)]
        target_means: [4]
        target_stds: [4]
    '''
    target_means = tf.constant(
        target_means, dtype=tf.float32)
    target_stds = tf.constant(
        target_stds, dtype=tf.float32)

    anchor_box = tf.cast(anchors, tf.float32)
    gt_box = tf.cast(gt_boxes, tf.float32)

    height = anchor_box[..., 2] - anchor_box[..., 0]
    width = anchor_box[..., 3] - anchor_box[..., 1]
    center_y = anchor_box[..., 0] + 0.5 * height
    center_x = anchor_box[..., 1] + 0.5 * width

    gt_height = gt_box[..., 2] - gt_box[..., 0]
    gt_width = gt_box[..., 3] - gt_box[..., 1]
    gt_center_y = gt_box[..., 0] + 0.5 * gt_height
    gt_center_x = gt_box[..., 1] + 0.5 * gt_width

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.math.log(gt_height / height)
    dw = tf.math.log(gt_width / width)

    delta = tf.stack([dy, dx, dh, dw], axis=-1)
    delta = (delta - target_means) / target_stds

    return delta
