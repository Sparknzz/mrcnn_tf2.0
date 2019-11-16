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
            Uses 0 padding to fill in unsed bbox deltas.
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

    # rpn bbox loss consists of only positive anchors
    pos_anchor_idx = tf.where(target_matchs, 1)

    pos_rpn_deltas = tf.gather_nd(rpn_deltas, pos_anchor_idx)

    # because rpn deltas is rpn output. shape is (Batch, anchors, 4)
    # however target deltas shape is [batch, num_rpn_deltas(256), 4],
    # we need to trim the target deltas to same as pos_rpn deltas

    batch_count = tf.reduce_sum(tf.cast(tf.where(target_matchs, 1), dtype=tf.int32), axis=1)  # [batch, 1]
    batch_size = target_deltas.shape.as_list()[0]
    # do batch trim to match the rpn_deltas
    target_deltas = batch_trim(target_deltas, batch_count, batch_size)  # [batch, pos_num, 4]

    loss = smooth_l1_loss(target_deltas, rpn_deltas)

    loss = tf.reduce_mean(loss) if tf.size(loss) > 0 else tf.constant(0.0)

    return loss
