from mrcnn.utils import *


class ProposalTarget(object):
    def __init__(self, config, target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rcnn_deltas=256,
                 positive_fraction=0.25,
                 mask_shape=[28, 28],
                 pos_iou_thr=0.5,
                 neg_iou_thr=0.5):
        self.config = config
        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rcnn_deltas = num_rcnn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.mask_shape = mask_shape

    def build_proposal_target(self, proposal_list, gt_boxes, gt_class_ids, gt_masks, img_metas):
        '''
        Generates detection targets for data. Subsamples proposals and
        generates target class IDs, bounding box deltas for each.

        Args
        ---
            proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in normalized coordinates.
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image coordinates.
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.
            gt_masks : [batch_size, num_gt_masks, (7*7)]
            img_metas: [batch_size, 14]

        Returns
        ---
            rois_list: list of [num_rois, (y1, x1, y2, x2)] in normalized coordinates
            rcnn_target_matchs_list: list of [num_rois]. Integer class IDs.
            rcnn_target_deltas_list: list of [num_positive_rois, (dy, dx, log(dh), log(dw))].

        Note that self.num_rcnn_deltas >= num_rois > num_positive_rois. And different
           data in one batch may have different num_rois and num_positive_rois.
        '''

        pad_shapes = get_batch_pad_shape(img_metas)

        rois_list = []
        rcnn_target_match_list = []
        rcnn_target_deltas_list = []
        rcnn_target_mask_list = []

        for i in range(img_metas.shape[0]):
            rois, target_match, target_deltas, target_masks = self._build_single_img_proposal_target(
                proposal_list[i], gt_boxes[i],
                gt_class_ids[i], gt_masks[i],
                pad_shapes[i])

            rois_list.append(rois)  # including pos/neg anchors
            rcnn_target_match_list.append(target_match)  # positive target deltas, and padding with zero for neg
            rcnn_target_deltas_list.append(target_deltas)  # positive target deltas, and padding with zero for neg
            rcnn_target_mask_list.append(target_masks)

        return rois_list, rcnn_target_match_list, rcnn_target_deltas_list, rcnn_target_mask_list

    def _build_single_img_proposal_target(self, proposals, gt_boxes, gt_class_ids, gt_masks, pad_shapes):
        '''
        proposals : [2000, 4]
        gt_boxes: [N, 4]
        gt_class_ids: [N]
        gt_masks: [height, width, MAX_GT_INSTANCES] of boolean type.
        pad_shapes: eg.[1024,1024]

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

        gt_masks = tf.gather(gt_masks, tf.where(non_zeros)[:, 0], axis=2,
                             name="trim_gt_masks")

        gt_class_ids = tf.boolean_mask(gt_class_ids, non_zeros)

        gt_boxes = tf.cast(gt_boxes, dtype=tf.float32) / tf.constant([H, W, H, W], dtype=tf.float32)

        # 2. calculate overlaps bet gt and proposals
        overlaps = compute_overlaps(proposals, gt_boxes)
        # returns [num_proposals, gt] matrix

        proposal_iou_max = tf.reduce_max(overlaps, axis=1)
        proposal_iou_argmax = tf.argmax(overlaps, axis=1)

        # 3. find out positive proposals
        pos_indices = tf.where(proposal_iou_max > self.pos_iou_thr)[:, 0]  # [N,]
        # 4. find out negative proposals according to ratio
        neg_indices = tf.where(proposal_iou_max < self.neg_iou_thr)[:, 0]  # [N,]

        pos_count = int(self.positive_fraction * self.num_rcnn_deltas)  # 256*0.25 = 64 at most 64
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
        roi_gt_box_assignment = tf.argmax(positive_overlaps, axis=1)
        roi_gt_box = tf.gather(gt_boxes, roi_gt_box_assignment)
        target_match = tf.gather(gt_class_ids, roi_gt_box_assignment)

        # target_matchs, target_deltas all get!!
        # target match and target deltas only calculated by positive anchor never be negative
        target_deltas = bbox2deltas(positive_rois, roi_gt_box, self.target_means, self.target_stds)

        ################################ for mask branch ########################################

        # 7. assign positive ROIs to GT masks
        # Permute masks to [N, height, width, 1]
        transposed_masks = tf.expand_dims(tf.transpose(gt_masks, [2, 0, 1]), -1)  # [N, H, W, C]
        # Pick the right mask for each ROI
        target_masks = tf.gather(transposed_masks, roi_gt_box_assignment)

        # Compute mask targets
        boxes = positive_rois

        if self.config.USE_MINI_MASK:
            # important!!! as the mini mask coordinate was resized to square
            # my understanding is that the mini mask is cropped from original img.
            # remember mini mask is just a part of img. however,
            # all gt or rois coordinates are based on img coordinates.
            # so you have to calculate the proportion of them.
            # which means the position should be normalized for small mask instead of whole img.
            # but the gt or rois are not resized!!!
            # so the roi need to be transformed to normalized position.
            y1, x1, y2, x2 = tf.split(positive_rois, 4, axis=1)
            gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(roi_gt_box, 4, axis=1)
            gt_h = gt_y2 - gt_y1
            gt_w = gt_x2 - gt_x1
            y1 = (y1 - gt_y1) / gt_h
            x1 = (x1 - gt_x1) / gt_w
            y2 = (y2 - gt_y1) / gt_h
            x2 = (x2 - gt_x1) / gt_w
            boxes = tf.concat([y1, x1, y2, x2], 1)

        box_ids = tf.range(0, tf.shape(target_masks)[0])
        target_masks = tf.image.crop_and_resize(tf.cast(target_masks, tf.float32), boxes,
                                                box_ids, self.mask_shape)

        # Remove the extra dimension from masks.
        # Shape is [num_rois, 28, 28]
        target_masks = tf.squeeze(target_masks, axis=3)

        # Threshold mask pixels at 0.5 to have GT masks be 0 or 1 to use with binary cross entropy loss.
        target_masks = tf.round(target_masks)

        # Append negative ROIs and pad bbox deltas and masks that are not used for negative ROIs with zeros.
        rois = tf.concat([positive_rois, negative_rois], axis=0)
        N = tf.shape(negative_rois)[0]
        P = tf.maximum(self.num_rcnn_deltas - tf.shape(rois)[0], 0)

        rois = tf.pad(rois, [(0, P), (0, 0)])
        target_match = tf.pad(target_match, [
            (0, N + P)])  # as we only count target match for positive before, so pad 0 to match with negative rois
        target_deltas = tf.pad(target_deltas, [(0, N + P), (0, 0)])

        target_masks = tf.pad(target_masks, [(0, N + P), (0, 0), (0, 0)])

        target_match = tf.stop_gradient(target_match)
        target_deltas = tf.stop_gradient(target_deltas)
        target_masks = tf.stop_gradient(target_masks)

        return rois, target_match, target_deltas, target_masks
