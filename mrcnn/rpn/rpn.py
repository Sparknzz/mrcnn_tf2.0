from mrcnn.rpn import anchor_generator, anchor_target
from mrcnn.utils import *
from mrcnn.loss import losses


class RPN(tf.keras.Model):
    def __init__(self, anchor_scales,
                 anchor_ratios=(0.5, 1, 2),
                 anchor_feature_strides=(4, 8, 16, 32, 64),
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 proposal_count=2000,
                 nms_threshold=0.7,
                 positive_fraction=0.33,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3, **kwargs):
        '''
        Network head of Region Proposal Network.

                                      / - rpn_cls (1x1 conv)
        input - rpn_conv (3x3 conv) -
                                      \ - rpn_reg (1x1 conv)

        Attributes
        ---
            anchor_scales: 1D array of anchor sizes in pixels. eg.[8, 16, 32, 64, 128]
            anchor_ratios: 1D array of anchor ratios of width/height. [0.5, 1, 2]
            anchor_feature_strides: Stride of the feature map relative to the image in pixels. [4, 8, 16, 32, 64]
            proposal_count: int. RPN proposals kept after non-maximum suppression.
            nms_threshold: float. Non-maximum suppression threshold to filter RPN proposals.
            target_means: [4] Bounding box refinement mean.
            target_stds: [4] Bounding box refinement standard deviation.
            num_rpn_deltas: int.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''

        super(RPN, self).__init__(**kwargs)

        self.proposal_count = proposal_count
        self.nms_threshold = nms_threshold
        self.target_means = target_means
        self.target_stds = target_stds

        self.generator = anchor_generator.AnchorGenerator(
            scales=anchor_scales,
            ratios=anchor_ratios,
            feature_strides=anchor_feature_strides)

        self.anchor_target = anchor_target.AnchorTarget(
            target_means=(0., 0., 0., 0.),
            target_stds=(0.1, 0.1, 0.2, 0.2),
            num_rpn_deltas=num_rpn_deltas,
            positive_fraction=positive_fraction,
            pos_iou_thr=pos_iou_thr,
            neg_iou_thr=neg_iou_thr)

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

        self.rpn_class_loss = losses.rpn_class_loss
        self.rpn_bbox_loss = losses.rpn_bbox_loss

    def call(self, feature_maps, **kwargs):
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

    def loss(self, rpn_class_logits, rpn_deltas, gt_boxes, gt_class_ids, img_metas):
        # NOTE RPN is used to regress the anchor and ground truth. the offset regression is deltas of (anchor, gt) - (rpn_deltas)
        # 1. anchor generator
        # todo anchor generated two times, other one for generate proposals
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)
        # 2. gt match
        rpn_target_matchs, rpn_target_deltas = self.anchor_target.build_targets(anchors, valid_flags, gt_boxes,
                                                                                gt_class_ids)

        rpn_class_loss = self.rpn_class_loss(rpn_class_logits, rpn_target_matchs)
        rpn_box_loss = self.rpn_bbox_loss(rpn_deltas, rpn_target_deltas, rpn_target_matchs)

        return rpn_class_loss, rpn_box_loss

    def get_proposals(self, rpn_probs, rpn_deltas, img_metas):
        """
        generate [N, (y1,x1,y2,x2)] proposals corresponding to image

        inputs:
            rpn_probs: [batch_size, num_anchors, (bg prob, fg prob)]
            rpn_deltas: [batch_size, num_anchors, (dy, dx, log(dh), log(dw))]
            img_metas: [batch_size, 11]
            img_meta_dict = dict({
                'id' : 1,
                'ori_shape': ori_shape,
                'img_shape': img_shape,
                'pad_shape': pad_shape,
            })
            with_probs: bool.

        :return:
         proposals_list: list of [num_proposals, (y1, x1, y2, x2)] in
                normalized coordinates if with_probs is False.
                Otherwise, the shape of proposals in proposals_list is
                [num_proposals, (y1, x1, y2, x2, score)]

        Note that num_proposals is no more than proposal_count. And different
           data in one batch may have different num_proposals.
        """
        # this including padded zero areas
        anchors, valid_flags = self.generator.generate_pyramid_anchors(img_metas)

        # [b, N, (background prob, foreground prob)], get anchor's foreground prob, [1, 369303]
        rpn_probs = rpn_probs[:, :, 1]

        pad_shape = get_batch_pad_shape(img_metas)

        proposals_list = [
            self._get_proposals_single_img(
                rpn_probs[i], rpn_deltas[i], anchors, valid_flags[i], pad_shape[i]) for i in
            range(img_metas.shape[0])]

        return proposals_list

    def _get_proposals_single_img(self, rpn_probs, rpn_deltas, anchors, valid_flags, pad_shape):
        '''
        Calculate proposals.

        Args
        ---
            rpn_probs: [num_anchors]
            rpn_deltas: [num_anchors, (dy, dx, log(dh), log(dw))]
            anchors: [num_anchors, (y1, x1, y2, x2)] anchors defined in
                pixel coordinates.
            valid_flags: [num_anchors] without padding area anchor's flag 0 or 1 for each position
            pad_shape: np.ndarray. [2]. (img_height, img_width)

        Returns
        ---
            proposals: [num_proposals, (y1, x1, y2, x2)] in normalized
                coordinates.
        '''

        H, W = pad_shape

        # filter invalid anchors, int => bool
        valid_flags = tf.cast(valid_flags, tf.bool)

        # boolean_mask returns 1-d array
        rpn_probs = tf.boolean_mask(rpn_probs, valid_flags)
        rpn_deltas = tf.boolean_mask(rpn_deltas, valid_flags)
        valid_anchors = tf.boolean_mask(anchors, valid_flags)

        # improve performance do prenms filtered lower probs
        pre_nms_limit = min(6000, valid_anchors.shape[0])
        ix = tf.nn.top_k(rpn_probs, pre_nms_limit, sorted=True).indices

        # obtain ix deltas
        rpn_probs = tf.gather(rpn_probs, ix)
        rpn_deltas = tf.gather(rpn_deltas, ix)
        valid_anchors = tf.gather(valid_anchors, ix)

        # get refinement anchors => [6000, 4]
        proposals = deltas2bbox(valid_anchors, rpn_deltas)

        # clipping to valid area [6000, 4] means even deleted padded area, there still are proposals outside of
        # padded img
        window = tf.constant([0., 0., H, W], dtype=tf.float32)
        proposals = self.proposal_clip(proposals, window=window)

        # Normalize y1,x1,y2,x2
        proposals = proposals / tf.constant([H, W, H, W], dtype=tf.float32)

        # do NMS to cut down 6000 to 2000 proposals
        indices = tf.image.non_max_suppression(proposals, rpn_probs, self.proposal_count, self.nms_threshold)
        # note gather and gather_nd  normally gather is axis 0 single dim search, but gather_nd supports multi dims
        # depends on indices dims
        proposals = tf.gather(proposals, indices)

        return proposals

    def proposal_clip(self, proposals, window):
        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(proposals, 4, axis=1)
        # clip
        y1 = tf.maximum(tf.minimum(y1, wy2), wy1)
        x1 = tf.maximum(tf.minimum(x1, wx2), wx1)
        y2 = tf.maximum(tf.minimum(y2, wy2), wy1)
        x2 = tf.maximum(tf.minimum(x2, wx2), wx1)

        # stack concat stack can increase a dim but concat cannot
        clipped = tf.concat([y1, x1, y2, x2], axis=1)
        clipped.set_shape((clipped.shape[0], 4))

        return clipped
