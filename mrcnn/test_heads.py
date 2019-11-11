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

        return proposals_list[0] # note bboxes might have NAN/INF coordinates wired??
