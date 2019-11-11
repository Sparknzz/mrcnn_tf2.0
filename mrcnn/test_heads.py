# this is the testing file, used to test all heads of mask rcnn
import tensorflow as tf
import numpy as np


class RPNTest:
    def simple_test_rpn(self, img, img_meta):
        '''
        Args
        ---
            imgs: np.ndarray. [height, width, channel] padded imgs
            img_metas: np.ndarray. [11]

        '''
        imgs = tf.Variable(np.expand_dims(img, 0), dtype=tf.float32)
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))

        x = self.backbone(imgs, training=False)
        x = self.neck(x, training=False)

        # all this are padded logits probs...
        rpn_class_logits, rpn_probs, rpn_deltas = self.rpn_head(x, training=False)

        # 1, rpn_anchors, 2
        proposals_list = self.rpn_head.get_proposals(rpn_probs, rpn_deltas, img_metas)

        return proposals_list[0]  # note bboxes might have NAN/INF coordinates wired??

    def simple_test_bboxes(self, img, img_meta, proposals):
        '''
        Args
        ---
            imgs: np.ndarray. [height, width, channel]
            img_meta: np.ndarray. [11]

            NOTE THAT proposals is a list of (2000, 4)
            SO REMEMBER rois_list means proposals

        '''
        imgs = tf.Variable(np.expand_dims(img, 0))
        img_metas = tf.Variable(np.expand_dims(img_meta, 0))
        rois_list = [tf.Variable(proposals)]

        x = self.backbone(imgs, training=False)
        P2, P3, P4, P5, _ = self.neck(x, training=False)

        rcnn_feature_maps = [P2, P3, P4, P5]

        pooled_regions_list = self.roi_align(
            (rois_list, rcnn_feature_maps, img_metas), training=False)

        rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list = \
            self.bbox_head(pooled_regions_list, training=False)

        detections_list = self.bbox_head.get_bboxes(
            rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas)

        return self._unmold_detections(detections_list, img_metas)[0]
