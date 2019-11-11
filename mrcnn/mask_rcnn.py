"""
Mask R-CNN
tf2.0 main Mask R-CNN model implemenetation.

Written by Yupeng
"""
import tensorflow as tf
from mrcnn.backbones import resnet
from mrcnn.fpn import fpn
from mrcnn.rpn import rpn
from mrcnn.test_heads import *
from mrcnn.roi_extractors import roi_align


class MaskRCNN(tf.keras.Model, RPNTest):
    def __init__(self, num_classes):
        super().__init__()
        self.NUM_CLASSES = num_classes
        self.ANCHOR_SCALES = [8, 16, 32, 64, 128]
        self.ANCHOR_RATIOS = [0.5, 1, 2]
        self.ANCHOR_FEATURE_STRIDES = [4, 8, 16, 32, 64]
        self.PRN_BATCH_SIZE = 256
        self.RPN_POS_FRAC = 0.5

        self.RPN_POS_IOU_THR = 0.7
        self.RPN_NEG_IOU_THR = 0.3

        self.RPN_TARGET_MEANS = [0, 0, 0, 0]
        self.RPN_TARGET_STDS = [0.1, 0.1, 0.2, 0.2]
        self.POOL_SIZE = 7

        # Modules
        self.backbone = resnet.ResNet(depth=101, name='res_net')
        self.neck = fpn.FPN()

        self.rpn_head = rpn.RPN(
            anchor_scales=self.ANCHOR_SCALES,
            anchor_ratios=self.ANCHOR_RATIOS,
            anchor_feature_strides=self.ANCHOR_FEATURE_STRIDES,
            num_rpn_deltas=self.PRN_BATCH_SIZE,
            # proposal_count=self.PRN_PROPOSAL_COUNT,
            # nms_threshold=self.PRN_NMS_THRESHOLD,
            target_means=self.RPN_TARGET_MEANS,
            target_stds=self.RPN_TARGET_STDS,
            positive_fraction=self.RPN_POS_FRAC,
            pos_iou_thr=self.RPN_POS_IOU_THR,
            neg_iou_thr=self.RPN_NEG_IOU_THR, )

        self.roi_align = roi_align.PyramidROIAlign(
            pool_shape=self.POOL_SIZE,
            name='pyramid_roi_align')

    def call(self, inputs, training=True):
        """
        :param inputs: [1, 1216, 1216, 3], [1, 11], [1, 14, 4], [1, 14]
        :param training:
        :return:
        """
        if training:  # training
            imgs, img_metas, gt_boxes, gt_class_ids = inputs
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
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(rpn_feature_maps, training=training)
        # [369303, 4] => [215169, 4], valid => [6000, 4], performance =>[2000, 4], NMS
        proposals_list = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, img_metas)
