from mrcnn.utils import *


class ProposalTarget(object):
    def __init__(self, target_means=(0., 0., 0., 0.),
                target_stds=(0.1, 0.1, 0.2, 0.2),
                num_rcnn_deltas=256,
                positive_fraction=0.25,
                pos_iou_thr=0.5,
                neg_iou_thr=0.5):
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_proposal_target(self, proposal_list, gt_boxes, gt_class_ids, img_metas):
        '''
        Generates detection targets for images. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.

        Args
        ---
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            img_metas: [batch_size, 11]

        Returns
        ---
            rois_list: list of [num_rois, (y1, x1, y2, x2)] in normalized coordinates
            rcnn_target_matchs_list: list of [num_rois]. Integer class IDs.
            rcnn_target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))].

        Note that self.num_rcnn_deltas >= num_rois > num_positive_rois. And different
           images in one batch may have different num_rois and num_positive_rois.
        '''

        pad_shapes = tf.cast(img_metas[:, 7:9], tf.int32)

        rois_list = []
        rcnn_target_match_list = []
        rcnn_target_deltas_list = []

        for i in range(img_metas.shape[0]):
            rois, target_match, target_deltas = self._build_single_img_proposal_target(proposal_list[i], gt_boxes[i],
                                                                                       gt_class_ids[i], pad_shapes[i])

            rois_list.append(rois)  # including pos/neg anchors
            rcnn_target_match_list.append(target_match)  # positive target deltas, and padding with zero for neg
            rcnn_target_deltas_list.append(target_deltas)  # positive target deltas, and padding with zero for neg

        return rois_list, rcnn_target_match_list, rcnn_target_deltas_list

    def _build_single_img_proposal_target(self, proposals, gt_boxes, gt_class_ids, pad_shapes):
        '''
        proposals : [2000, 4]
        gt_boxes: [N, 4]
        gt_class_ids: [N]
        pad_shapes: eg.[1216,1216]

        return :
            this is only in training process, the 256 is not certain value. maybe 134....
            rois: [N, 4]
            target_deltas: [N, 4]
            target_matches: [N]
        '''
        H, W = pad_shapes

        # 1. trim 0 gt_boxes
        # non_zeros : boolean value of gt_boxes
        # as gt_boxes as inputs, it should be batch size, so it has to be the padding gts otherwise it cannot be a batch
        gt_boxes, non_zeros = trim_zeros(gt_boxes)

        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)

        gt_boxes = gt_boxes / tf.constant([H, W, H, W], dtype=tf.float32)

        # 2. calculate overlaps bet gt and proposals
        overlaps = compute_overlaps(proposals, gt_boxes)
        # returns [num_proposals, gt] matrix

        proposal_iou_max = tf.reduce_max(overlaps, axis=1)
        proposal_iou_argmax = tf.argmax(overlaps, axis=1)

        # 3. find out positive proposals
        pos_indices = tf.where(proposal_iou_max > self.pos_iou_thr)[:, 0]  # [N,]
        # 4. find out negative proposals according to ratio
        neg_indices = tf.where(proposal_iou_max < self.neg_iou_thr)[:, 0]  # [N,]

        pos_count = self.positive_fraction * self.num_rcnn_deltas  # 256*0.25 = 64 at most 64
        pos_indices = tf.random.shuffle(pos_indices)[:pos_count]
        pos_count = tf.shape(pos_indices)[0]

        r = 1.0 / self.positive_fraction
        neg_count = tf.cast(r * tf.cast(pos_count, tf.float32), tf.int32) - pos_count
        neg_indices = tf.random.shuffle(neg_indices)[:neg_count]  # 256 - 64 = 192  at most 192

        # 5. gather selected rois based on removed redundant pos/neg indices
        positive_rois = tf.gather(proposals, pos_indices)
        negative_rois = tf.gather(proposals, neg_indices)

        # 6. assign positive rois to gt boxes
        positive_overlaps = tf.gather(overlaps, pos_indices)
        roi_gt_box_assigment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_box = tf.gather(gt_boxes, roi_gt_box_assigment)
        target_match = tf.gather(gt_class_ids, roi_gt_box_assigment)
        # target_matchs, target_deltas all get!!
        #!! important here, target match and target deltas only calculated by positive anchor never be negative
        target_deltas = bbox2deltas(positive_rois, roi_gt_box, self.target_means, self.target_stds)

        rois = tf.concat([positive_rois, negative_rois], axis=0)

        N = tf.shape(negative_rois)[0]
        target_match = tf.pad(target_match, [
            (0, N)])  # as we only count target match for positive before, so pad 0 to match with negative rois

        target_match = tf.stop_gradient(target_match)
        target_deltas = tf.stop_gradient(target_deltas)

        return rois, target_match, target_deltas
