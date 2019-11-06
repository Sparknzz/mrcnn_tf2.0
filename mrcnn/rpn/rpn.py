import tensorflow as tf


class RPN(tf.keras.Model):
    def __init__(self, anchors_per_location, anchor_stride,
                 anchor_ratios=(0.5, 1, 2),
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 proposal_count=2000,
                 nms_threshold=0.7,
                 positive_fraction=0.33,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3, ):
        '''
        Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels.
            anchor_ratios: 1D array of anchor ratios of width/height.
            anchor_feature_strides: Stride of the feature map relative
                to the image in pixels.
            proposal_count: int. RPN proposals kept after non-maximum
                suppression.
            nms_threshold: float. Non-maximum suppression threshold to
                filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''
        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds

        self.anchors = anchor_generator.AnchorGenerator(
            scales=anchor_scales,
            ratios=anchor_ratios,
            feature_strides=anchor_feature_strides)










        # Shared convolutional base of the RPN
        self.rpn_conv_shared = tf.keras.layers.Conv2D(512, (3, 3), padding='same',
                                                      kernel_initializer='he_normal',
                                                      name='rpn_conv_shared')

        self.rpn_class_raw = tf.keras.layers.Conv2D(len(anchor_ratios) * 2, (1, 1),
                                                    kernel_initializer='he_normal',
                                                    name='rpn_class_raw')

        self.rpn_delta_pred = tf.keras.layers.Conv2D(len(anchor_ratios) * 4, (1, 1),
                                                     kernel_initializer='he_normal',
                                                     name='rpn_bbox_pred')

    def call(self, feature_maps):
        '''
        Args
        inputs: [batch_size, feat_map_height, feat_map_width, channels]
        one level of pyramid feat-maps.

        Returns
        ---
            rpn_class_logits: [batch_size, num_anchors, 2]
            rpn_probs: [batch_size, num_anchors, 2]
            rpn_deltas: [batch_size, num_anchors, 4]
        '''
        layer_outputs = []

        for feat in feature_maps:
            """
            (1, 304, 304, 256)
            (1, 152, 152, 256)
            (1, 76, 76, 256)
            (1, 38, 38, 256)
            (1, 19, 19, 256)
            rpn_class_raw: (1, 304, 304, 6)
            rpn_class_logits: (1, 277248, 2)
            rpn_delta_pred: (1, 304, 304, 12)
            rpn_deltas: (1, 277248, 4)
            rpn_class_raw: (1, 152, 152, 6)
            rpn_class_logits: (1, 69312, 2)
            rpn_delta_pred: (1, 152, 152, 12)
            rpn_deltas: (1, 69312, 4)
            rpn_class_raw: (1, 76, 76, 6)
            rpn_class_logits: (1, 17328, 2)
            rpn_delta_pred: (1, 76, 76, 12)
            rpn_deltas: (1, 17328, 4)
            rpn_class_raw: (1, 38, 38, 6)
            rpn_class_logits: (1, 4332, 2)
            rpn_delta_pred: (1, 38, 38, 12)
            rpn_deltas: (1, 4332, 4)
            rpn_class_raw: (1, 19, 19, 6)
            rpn_class_logits: (1, 1083, 2)
            rpn_delta_pred: (1, 19, 19, 12)
            rpn_deltas: (1, 1083, 4)
            """
            shared = self.rpn_conv_shared(feat)
            shared = tf.nn.relu(shared)

            x = self.rpn_class_raw(shared)
            rpn_class_logits = tf.reshape(x, [tf.shape(x)[0], -1, 2])
            rpn_probs = tf.nn.softmax(rpn_class_logits)  # batch, H*W, 2

            x = self.rpn_delta_pred(shared)
            rpn_deltas = tf.reshape(x, [tf.shape(x)[0], -1, 4])  # batch, H*W, 2

            layer_outputs.append([rpn_class_logits, rpn_probs, rpn_deltas])

        outputs = list(zip(*layer_outputs))
        outputs = [tf.concat(list(o), axis=1) for o in outputs]

        rpn_class_logits, rpn_probs, rpn_deltas = outputs

        return rpn_class_logits, rpn_probs, rpn_deltas

    def loss(self):
        pass