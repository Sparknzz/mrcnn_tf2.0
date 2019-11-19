import tensorflow as tf


def rpn_class_loss(rpn_class_logits, target_matchs):
    '''RPN anchor classifier loss.

    Args
    ---
        target_matchs: [batch_size, num_anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_class_logits: [batch_size, num_anchors, 2]. RPN classifier logits for FG/BG.
    '''
    # convert -1, +1 value to 0, 1
    anchor_class = tf.cast(tf.equal(target_matchs, 1), dtype=tf.int32)
    # Positive and Negative anchors contribute to the loss,
    # but neutral anchors (match value = 0) don't.
    indices = tf.where(tf.not_equal(target_matchs, 0))
    # Pick rows that contribute to the loss and filter out the rest.
    rpn_class_logits = tf.gather_nd(rpn_class_logits, indices)

    anchor_class = tf.gather_nd(anchor_class, indices)

    num_classes = rpn_class_logits.shape[-1]

    loss = tf.keras.losses.categorical_crossentropy(tf.one_hot(anchor_class, depth=num_classes), rpn_class_logits,
                                                    from_logits=True)

    return loss


def smooth_l1_loss(y_true, y_pred):
    '''Implements Smooth-L1 loss.

    当预测值与目标值相差很大时，L2 Loss的梯度为(x-t)，容易产生梯度爆炸
    L1 Loss的梯度为常数，通过使用Smooth L1 Loss，在预测值与目标值相差较大时，由L2 Loss转为L1 Loss可以防止梯度爆炸。
    Args
    ---
        y_true and y_pred are typically: [N, 4], but could be any shape.

    return: loss [N, 4]
    '''

    diff = tf.abs(y_true - y_pred)
    less_than_one = tf.cast(tf.less(diff, 1), dtype=tf.float32)

    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss


def rpn_bbox_loss(rpn_deltas, target_deltas, target_matchs):
    '''
    Return the RPN bounding box loss graph.
    Args
    ---
        target_deltas: [batch, num_rpn_deltas, (dy, dx, log(dh), log(dw))].
            Uses 0 padding to fill in unsed rcnn deltas.
            IMPORTANT: target_deltas is only for pos anchors, and because we padding it when we generate them.
            So need to be trimmed
        target_matchs: [batch, anchors]. Anchor match type. 1=positive,
            -1=negative, 0=neutral anchor.
        rpn_deltas: [batch, anchors, (dy, dx, log(dh), log(dw))]
    '''

    def batch_trim(target_deltas, batch_count, batch_size):
        outputs = []

        for i in range(batch_size):
            outputs.append(target_deltas[i, :batch_count[i]])

        return tf.concat(outputs, axis=0)

    # rpn rcnn loss consists of only positive anchors
    pos_anchor_idx = tf.where(target_matchs, 1)

    pos_rpn_deltas = tf.gather_nd(rpn_deltas, pos_anchor_idx)

    # because rpn deltas is rpn output. shape is (Batch, anchors, 4)
    # however target deltas shape is [batch, num_rpn_deltas(256), 4], and only few positive, we need to trim the zeros
    # we need to trim the target deltas to same as pos_rpn deltas

    batch_count = tf.reduce_sum(tf.cast(tf.where(target_matchs, 1), dtype=tf.int32), axis=1)  # [batch, 1]
    batch_size = target_deltas.shape.as_list()[0]
    # do batch trim to match the rpn_deltas
    target_deltas = batch_trim(target_deltas, batch_count, batch_size)  # [batch, pos_num, 4]

    loss = smooth_l1_loss(target_deltas, rpn_deltas)

    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss


def rcnn_bbox_loss(target_deltas_list, target_matchs_list, rcnn_deltas_list):
    '''Loss for Faster R-CNN bounding box refinement.
    note: this is only used in training, so target_matches_list depends on the target, maybe 256 default.
    but actually it is not certain value. maybe 192 or 134  etc. including pos and neg.

    rcnn_deltas_list: either training or inference, it would be there. in training stage, same as target.
    in inference it would be 2000 always. [2000 * num_class, 4]
    so have to figure out which pooled_roi and which class_id it coresponding.

    Args
    ---
        target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))]
        target_matchs_list: list of [num_rois]. Integer class IDs.
        rcnn_deltas_list: list of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
    '''
    target_deltas = tf.concat(target_deltas_list, axis=0)
    target_matchs = tf.concat(target_matchs_list, axis=0)
    rcnn_deltas = tf.concat(rcnn_deltas_list, axis=0)

    # Only positive ROIs contribute to the loss. And only the right class_id of each ROI. Get their indicies.
    positive_roi_indices = tf.where(target_matchs > 0)[:, 0]  # list of indices[]
    positive_roi_class_ids = tf.cast(tf.gather(target_matchs, positive_roi_indices), dtype=tf.int32)
    indices = tf.stack([positive_roi_indices, positive_roi_class_ids], axis=1)

    # Gather the deltas (predicted and true) that contribute to loss
    rcnn_deltas = tf.gather_nd(rcnn_deltas, indices)

    # smooth l1 loss
    loss = smooth_l1_loss(target_deltas, rcnn_deltas)

    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss


def rcnn_class_loss(rcnn_target_matchs_list, rcnn_class_logits_list):
    '''
    rcnn_target_matchs_list : list of [num_rois]. Integer class IDs. Uses zero padding to fill in the array.
    rcnn_class_logits_list : list of [num_rois, num_classes]
    '''

    class_ids = tf.concat(rcnn_target_matchs_list, axis=0)
    class_logits = tf.concat(rcnn_class_logits_list, axis=0)
    class_ids = tf.cast(class_ids, tf.int32)

    num_classes = rcnn_class_logits_list.shape[-1]

    loss = tf.keras.losses.categorical_crossentropy(tf.one_hot(class_ids, depth=num_classes), class_logits,
                                                    from_logits=True)

    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss
