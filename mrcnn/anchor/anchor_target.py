from mrcnn.utils import *

"""
for every generated anchors boxes:
create its rpn_target_matchs and rpn_target_deltas
rpn_target_matchs: 
rpn_target_deltas: 
which is used to train RPN network.
"""


class AnchorTarget:
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''
        Compute regression and classification targets for anchors.

        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''

        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, gt_boxes, gt_class_ids):
        '''
        Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image
                coordinates. batch_size = 1 usually
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.

        Returns
        ---
            rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))]
                Anchor bbox deltas.
        '''
        rpn_target_matches = []
        rpn_target_deltas = []

        num_images = gt_boxes.shape[0]  # namely, batch , 1

        for i in range(num_images):
            self._build_single_target(anchors, gt_boxes[i], gt_class_ids[i])

        pass

    def _build_single_target(self, anchors, gt_boxes, gt_class_ids):
        '''Compute targets per instance.
        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)]
            valid_flags: [num_anchors]
            gt_class_ids: [num_gt_boxes]
            gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

        Returns
        ---
            target_matchs: [num_anchors]
            target_deltas: [num_rpn_deltas, (dy, dx, log(dh), log(dw))]
        '''

        target_matches = tf.zeros(anchors.shape[0], dtype=tf.int32)  # each anchor have a label

        # Compute overlaps [num_anchors, num_gt_boxes]
        overlaps = compute_overlaps(anchors, gt_boxes)

        # Match anchors to GT Boxes
        # If an anchor overlaps ANY GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps ALL GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        neg_values = tf.constant([0, -1])
        pos_values = tf.constant([0, 1])

        # get closest gt for each anchor
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)
        # get closest gt scores for each anchor
        anchor_iou_max = tf.reduce_max(overlaps, axis=1)

        # 1. set negative anchors, will be overwritten below
        target_matches = tf.where(anchor_iou_max < self.neg_iou_thr, -tf.ones(anchors.shape[0], dtype=tf.int32),
                                  target_matches)

        # 2.filter invalid anchors
        # todo

        # 3. if an anchor overlap with any GT box with IOU>0.7, marked as foreground
        target_matches = tf.where(anchor_iou_max > self.pos_iou_thr, tf.ones(anchors.shape[0], dtype=tf.int32),
                                  target_matches)

        # 4. Set an anchor for each GT box (regardless of IoU value).
        # update corresponding value=>1 for GT boxes' closest boxes
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        target_matches = tf.compat.v1.scatter_update(target_matches, gt_iou_argmax, 1)

        # 5. Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(target_matches, 1))  # [N_pos_anchors, 1], [15, 1]
        ids = tf.squeeze(ids, 1)  # [15]

        # 6. cal diff for pos ids and set ids
        diff = ids.shape.as_list()[0] - int(self.num_rpn_deltas * self.positive_fraction)
        if diff > 0:  # means the redundant pos anchors
            ids = tf.random.shuffle(ids)[:diff]
            target_matches = tf.compat.v1.scatter_update(target_matches, ids, 0)

        # 7. same for negative proposals
        ids = tf.where(tf.equal(target_matches, -1))
        ids = tf.squeeze(ids, 1)

        diff = ids.shape.as_list()[0] - (self.num_rpn_deltas - tf.reduce_sum(
            tf.cast(tf.equal(target_matches, 1), dtype=tf.int32)))

        if diff > 0:  # means too many negative anchors
            ids = tf.random.shuffle(ids)[:diff]
            target_matches = tf.compat.v1.scatter_update(target_matches, ids, 0)

        # 8. for pos anchors, compute shifts and scales to transform them to match the corresponding GT boxes.
        ids = tf.where(tf.equal(target_matches, 1))
        anchors = tf.gather_nd(anchors, ids)

        anchor_idx = tf.gather_nd(anchor_iou_argmax, ids)  # closet gt boxes for all anchors
        gt = tf.gather(gt_boxes, anchor_idx)  # get closed gt boxes coordinates

        target_deltas = bbox2deltas(anchors, gt, self.target_means, self.target_stds)
        # target means all selected anchors

        # padding targets
        padding = tf.maximum(self.num_rpn_deltas - tf.shape(target_deltas)[0], 0)
        target_deltas = tf.pad(target_deltas, [[0, padding], [0, 0]])

        return target_matches, target_deltas
