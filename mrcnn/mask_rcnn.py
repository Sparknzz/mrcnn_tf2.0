"""
Mask R-CNN
tf2.0 main Mask R-CNN model implemenetation.

Written by Yupeng
"""
from mrcnn.backbones import resnet
from mrcnn.fpn import fpn
from mrcnn.roi_extractors import roi_align
from mrcnn.rpn import rpn
from mrcnn.test_heads import *
from mrcnn.rcnn import bbox_head, detection_target, mask_head
import tensorflow as tf

class MaskRCNN(tf.keras.Model, RPNTest):
    def __init__(self, num_classes, config):
        super().__init__()
        # global attributes
        self.NUM_CLASSES = num_classes  # including background
        self.RPN_TARGET_MEANS = [0, 0, 0, 0]
        self.RPN_TARGET_STDS = [0.1, 0.1, 0.2, 0.2]
        self.ANCHOR_FEATURE_STRIDES = [4, 8, 16, 32, 64]

        # first stage attributes
        self.ANCHOR_SCALES = config.RPN_ANCHOR_SCALES
        self.ANCHOR_RATIOS = config.RPN_ANCHOR_RATIOS

        # first stage nms anchors, after nms is 2000 anchors
        self.PRN_NMS_THRESHOLD = 0.5
        self.PRN_PROPOSAL_COUNT = 2000

        # stage 1 training part attrs
        self.PRN_BATCH_SIZE = 256
        self.RPN_POS_FRAC = 0.5
        self.RPN_POS_IOU_THR = 0.7
        self.RPN_NEG_IOU_THR = 0.3

        # roi attributes
        self.POOL_SIZE = (7, 7)

        # second stage attributs
        self.RCNN_BATCH_SIZE = 256
        self.RCNN_POS_FRAC = 0.25
        self.RCNN_POS_IOU_THR = 0.5
        self.RCNN_NEG_IOU_THR = 0.5

        # Modules
        self.backbone = resnet.ResNet(depth=101, name='res_net')
        self.neck = fpn.FPN()

        # stage 1
        self.rpn_head = rpn.RPN(
            anchor_scales=self.ANCHOR_SCALES,
            anchor_ratios=self.ANCHOR_RATIOS,
            anchor_feature_strides=self.ANCHOR_FEATURE_STRIDES,
            num_rpn_deltas=self.PRN_BATCH_SIZE,
            proposal_count=self.PRN_PROPOSAL_COUNT,
            nms_threshold=self.PRN_NMS_THRESHOLD,
            target_means=self.RPN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            positive_fraction=self.RPN_POS_FRAC,
            pos_iou_thr=self.RPN_POS_IOU_THR,
            neg_iou_thr=self.RPN_NEG_IOU_THR, )

        self.roi_align = roi_align.PyramidROIAlign(
            pool_shape=self.POOL_SIZE,
            name='pyramid_roi_align')

        # stage 2 rcnn
        self.bbox_target = detection_target.ProposalTarget(
            config=config,
            target_means=self.RPN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            num_rcnn_deltas=self.RCNN_BATCH_SIZE,
            positive_fraction=self.RCNN_POS_FRAC,
            pos_iou_thr=self.RCNN_POS_IOU_THR,
            neg_iou_thr=self.RCNN_NEG_IOU_THR)

        # NOTE this is implemented by put all rois together to do batch prediction, will split at the last
        self.bbox_head = bbox_head.BBoxHead(
            self.NUM_CLASSES,
            pool_size=(7, 7),
            target_means=(0., 0., 0., 0.),
            target_stds=(0.1, 0.1, 0.2, 0.2),
            min_confidence=0.7,
            nms_threshold=0.3,
            max_instances=100, )

        # stage 2 mask branch
        self.mask_head = mask_head.MaskHead(self.NUM_CLASSES, )

    def call(self, inputs, training=True):
        """
        :param inputs: [1, 1216, 1216, 3], [1, 11], [1, 14, 4], [1, 14]
        :param training: imgs, img_metas, gt_boxes, gt_class_ids
        :return:
        """
        if training:  # training
            imgs, img_metas, gt_class_ids, gt_boxes, gt_masks = inputs
        else:  # inference
            imgs, img_metas = inputs

        # [1, 304, 304, 256] => [1, 152, 152, 512] => [1,76,76,1024] => [1,38,38,2048]
        C2, C3, C4, C5 = self.backbone(imgs, training=training)
        # [1, 304, 304, 256] <= [1, 152, 152, 256]<=[1,76,76,256]<=[1,38,38,256]=>[1,19,19,256]
        P2, P3, P4, P5, P6 = self.neck([C2, C3, C4, C5], training=training)

        # Note that P6 is used in RPN, but not in the classifier heads.
        rpn_feature_maps = [P2, P3, P4, P5, P6]
        rcnn_feature_maps = [P2, P3, P4, P5]

        # [1, 369303, 2] [1, 369303, 2], [1, 369303, 4], includes all anchor on pyramid level of features
        # stage 1 attributes
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)

        # stage 2 attributes
        # when the logic comes to here, that means the proposal refinement is ok. proposal is ready to use
        # returns the normalized coordinates y1, x1, y2, x2
        # NOTE proposals is for stage 2, no relationship with training stage 1 proposals is all foreground anchors.
        proposals_list = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, img_metas)  # 2000
        # todo note here, interesting, 2000 proposals generated, when training rpn. we only need 256.
        #  however, 2000 still used to train stage2. because coding issue, if training, then 2000 will do a selection.

        ###########################################  core  ###################################################
        if training:
            # get target value for these proposal target label and target delta
            # NOTE IMPORTANT HERE IS PREPARE TRAINING SECOND STAGE
            # NOTE HERE rois_list is not certain batch it maybe 192, 134 depends on positive anchors value
            # and controlled by 1:3 for pos and neg
            rois_list, rcnn_target_matchs_list, rcnn_target_deltas_list, rcnn_target_mask_list = \
                self.bbox_target.build_proposal_target(
                    proposals_list, gt_boxes, gt_class_ids, gt_masks, img_metas)

        else:
            rois_list = proposals_list
        #######################################################################################################

        # rois_list only contains coordinates, rcnn_feature_maps save the 4 features data
        pooled_regions_list = self.roi_align(
            (rois_list, rcnn_feature_maps, img_metas), training=training)

        #############################################  stage 2  ##############################################
        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = self.bbox_head(pooled_regions_list,
                                                                                   training=training)
        # mask branch
        mrcnn_mask_list = self.mask_head(pooled_regions_list)

        if training:
            # note for rpn training, the rpn_deltas is all anchors deltas.
            rpn_class_loss, rpn_bbox_loss = self.rpn_head.loss(rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids,
                                                               img_metas)

            rcnn_class_loss, rcnn_bbox_loss = self.bbox_head.loss(rcnn_class_logits_list, rcnn_deltas_list,
                                                                  rcnn_target_matchs_list,
                                                                  rcnn_target_deltas_list)

            rcnn_mask_loss = self.mask_head.loss(rcnn_target_mask_list, rcnn_target_matchs_list, mrcnn_mask_list)

            return [rpn_class_loss, rpn_bbox_loss,
                    rcnn_class_loss, rcnn_bbox_loss, rcnn_mask_loss]

        else:
            detections_list = self.bbox_head.get_bboxes(
                rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)

            return detections_list
