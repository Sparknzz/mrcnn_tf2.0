from mrcnn.utils import *

"""
for every generated anchors boxes:
create its rpn_target_matchs and rpn_target_deltas
NOTE : VERY IMPORTANT you might think when do the regression, it should be proposals and gts.
but the thing is only anchors will do all calculate thing. this class is used for calculate deltas for all anchors and gts.
only returns 256 deltas for rpn training
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

    def build_targets(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''
        Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates. in traing
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image
                coordinates. batch_size = 1 usually
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.

        Returns
        ---
            rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))]
                Anchor rcnn deltas.
        '''
        rpn_target_matches = []
        rpn_target_deltas = []

        num_images = gt_boxes.shape[0]  # namely, batch , 1

        for i in range(num_images):
            target_match, target_delta = self._build_single_img_target(
                anchors, valid_flags[i], gt_boxes[i], gt_class_ids[i])

            rpn_target_matches.append(target_match)
            rpn_target_deltas.append(target_delta)

        rpn_target_matches = tf.stack(rpn_target_matches, axis=0)
        rpn_target_deltas = tf.stack(rpn_target_deltas, axis=0)

        rpn_target_matchs = tf.stop_gradient(rpn_target_matches)
        rpn_target_deltas = tf.stop_gradient(rpn_target_deltas)

        return rpn_target_matchs, rpn_target_deltas

    def _build_single_img_target(self, anchors, valid_flags, gt_boxes, gt_class_ids):
        '''Compute targets per instance.

         Args
         ---
             anchors: [num_anchors, (y1, x1, y2, x2)]
             valid_flags: [num_anchors]
             gt_class_ids: [num_gt_boxes]
             gt_boxes: [num_gt_boxes, (y1, x1, y2, x2)]

         Returns
         ---
             target_matchs: [num_anchors] 256? always? if not enough positive padding it to 256
             target_deltas: [num_rpn_deltas, (dy, dx, log(dh), log(dw))]
         '''

        gt_boxes, _ = trim_zeros(gt_boxes)  # remove padded zero boxes, [new_N, 4]

        # compute overlaps gt and anchors
        overlaps = compute_overlaps(anchors, gt_boxes)

        target_match = tf.zeros(anchors.shape[0], dtype=tf.int32)

        # Match anchors to GT Boxes
        # If an anchor overlaps ANY GT box with IoU >= 0.7 then it's positive.
        # If an anchor overlaps ALL GT box with IoU < 0.3 then it's negative.
        # Neutral anchors are those that don't match the conditions above,
        # and they don't influence the loss function.
        # However, don't keep any GT box unmatched (rare, but happens). Instead,
        # match it to the closest anchor (even if its max IoU is < 0.3).

        anchor_iou_max = tf.reduce_max(overlaps, axis=1)
        anchor_iou_argmax = tf.argmax(overlaps, axis=1)

        # set negative anchors -1
        target_match = tf.where(anchor_iou_max < self.neg_iou_thr, -tf.ones(anchors.shape[0], dtype=tf.int32),
                                target_match)

        # filter invalid anchors
        target_match = tf.where(tf.equal(valid_flags, 1), target_match, tf.zeros(anchors.shape[0], dtype=tf.int32))

        # set pos anchors as foreground
        target_match = tf.where(anchor_iou_max > self.pos_iou_thr, tf.ones(anchors.shape[0], dtype=tf.int32),
                                target_match)

        # set an anchor for gt regardless iou
        gt_iou_argmax = tf.argmax(overlaps, axis=0)
        target_match = tf.compat.v1.scatter_update(target_match, gt_iou_argmax, 1)

        # Subsample to balance positive and negative anchors
        # Don't let positives be more than half the anchors
        ids = tf.where(tf.equal(target_match, 1))
        ids = tf.squeeze(ids, -1)

        extra = ids.shape.as_list[0] - int(self.positive_fraction * self.num_rpn_deltas)
        if extra > 0:
            # reset the extra number to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_match = tf.compat.v1.scatter_update(target_match, ids, 0)

        # same for negative
        ids = tf.where(tf.equal(target_match, -1))
        ids = tf.squeeze(ids, -1)
        extra = ids.shape.as_list[0] - int(
            self.num_rpn_deltas - tf.reduce_sum(tf.cast(tf.equal(target_match, 1), dtype=tf.int32)))

        if extra > 0:
            # set to neutral
            ids = tf.random.shuffle(ids)[:extra]
            target_match = tf.compat.v1.scatter_update(target_match, ids, 0)

        # since we only need 256 anchors, and it had better contains half positive anchors, and harlf neg .
        # For positive anchors, compute shift and scale needed to transform them
        # to match the corresponding GT boxes.
        ids = tf.where(tf.equal(target_match, 1))
        pos_anchor = tf.gather_nd(anchors, ids)
        gt_idx = tf.gather_nd(anchor_iou_argmax, ids)  # closed gt boxes index for 369303 anchors

        gt = tf.gather_nd(gt_boxes, gt_idx)  # get closed gt boxes coordinates for ids=15

        # calculate deltas for gt and anchor
        target_deltas = bbox2deltas(pos_anchor, gt)

        # target_deltas: [15, (dy,dx,logw,logh)]?
        padding = tf.maximum(self.num_rpn_deltas - tf.shape(target_deltas)[0], 0)  # 256-15
        target_deltas = tf.pad(target_deltas, [(0, padding), (0, 0)])  # padding to [256,4], last padding 0

        return target_match, target_deltas
