from tensorflow.keras import layers

from mrcnn.loss import losses
from mrcnn.utils import *


class BBoxHead(tf.keras.Model):
    def __init__(self, num_classes,
                 pool_size=(7, 7),
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 min_confidence=0.7,
                 nms_threshold=0.3,
                 max_instances=100,
                 **kwags):
        super(BBoxHead, self).__init__(**kwags)

        self.num_classes = num_classes
        self.pool_size = tuple(pool_size)
        self.target_means = target_means
        self.target_stds = target_stds
        self.min_confidence = min_confidence
        self.nms_threshold = nms_threshold
        self.max_instances = max_instances

        self.rcnn_class_conv1 = layers.Conv2D(1024, self.pool_size,
                                              padding='valid', name='rcnn_class_conv1')

        self.rcnn_class_bn1 = layers.BatchNormalization(name='rcnn_class_bn1')

        self.rcnn_class_conv2 = layers.Conv2D(1024, (1, 1),
                                              name='rcnn_class_conv2')

        self.rcnn_class_bn2 = layers.BatchNormalization(name='rcnn_class_bn2')

        self.rcnn_class_logits = layers.Dense(num_classes, name='rcnn_class_logits')

        self.rcnn_delta_fc = layers.Dense(num_classes * 4, name='rcnn_bbox_fc')

        self.rcnn_class_loss = losses.rcnn_class_loss
        self.rcnn_bbox_loss = losses.rcnn_bbox_loss

    def call(self, pooled_rois_list, training=True):
        '''
           Args
           ---
               pooled_rois_list: List of [num_rois, pool_size, pool_size, channels]

           Returns
           ---
               rcnn_class_logits_list: List of [num_rois, num_classes]
               rcnn_probs_list: List of [num_rois, num_classes]
               rcnn_deltas_list: List of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
        '''

        num_pooled_rois_list = [pooled_rois.shape[0] for pooled_rois in pooled_rois_list]
        pooled_rois = tf.concat(pooled_rois_list, axis=0)

        x = self.rcnn_class_conv1(pooled_rois)
        x = self.rcnn_class_bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.rcnn_class_conv2(x)
        x = self.rcnn_class_bn2(x, training=training)
        x = tf.nn.relu(x)

        # x is [n, 1, 1, 1024]
        x = tf.squeeze(tf.squeeze(x, 2), 1)

        logits = self.rcnn_class_logits(x)  # [N,num_classes]
        probs = tf.nn.softmax(logits, axis=-1)

        deltas = self.rcnn_delta_fc(x)  # [N, 4*num_classes]
        deltas = tf.reshape(deltas, (-1, self.num_classes, 4))

        rcnn_class_logits_list = tf.split(logits, num_pooled_rois_list, axis=0)
        rcnn_probs_list = tf.split(probs, num_pooled_rois_list, axis=0)
        rcnn_deltas_list = tf.split(deltas, num_pooled_rois_list, axis=0)

        return rcnn_class_logits_list, rcnn_probs_list, rcnn_deltas_list

    def loss(self, rcnn_class_logits_list, rcnn_deltas_list, rcnn_target_matchs_list, rcnn_target_deltas_list):
        '''
            rcnn_class_logits_list: list of [batch, num_classes]
            rcnn_deltas_list: list of [batch, deltas]

            return bbox regression  bbox classification
        '''

        rcnn_class_loss = self.rcnn_class_loss(rcnn_target_matchs_list, rcnn_class_logits_list)
        rcnn_bbox_loss = self.rcnn_bbox_loss(rcnn_target_deltas_list, rcnn_target_matchs_list, rcnn_deltas_list)

        return rcnn_class_loss, rcnn_bbox_loss

    def get_bboxes(self, rcnn_probs_list, rcnn_deltas_list, rois_list, img_metas):
        '''
        Args
        ---
            rcnn_probs_list: List of [num_rois, num_classes]
            rcnn_deltas_list: List of [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois_list: List of [num_rois, (y1, x1, y2, x2)]
            img_meta_list: [batch_size, 11]

        decode process.

        Returns
        ---
            detections_list: List of [num_detections, (y1, x1, y2, x2, class_id, score)]
                coordinates are in pixel coordinates.
        '''

        pad_shape = tf.cast(img_metas[:, 7:9], dtype=tf.int32).numpy()

        detection_list = [
            self._get_single_img_detecion_list(rcnn_probs_list[i], rcnn_deltas_list[i], rois_list[i], pad_shape[i]) for
            i in range(img_metas.shape[0])]

        return detection_list

    def _get_single_img_detecion_list(self, rcnn_probs, rcnn_deltas, rois, img_shape):
        '''
        Args
        ---
            rcnn_probs: [num_rois, num_classes]
            rcnn_deltas: [num_rois, num_classes, (dy, dx, log(dh), log(dw))]
            rois: [num_rois, (y1, x1, y2, x2)]
            img_shape: np.ndarray. [2]. (img_height, img_width)
        '''

        H, W = img_shape
        # cls ids per roi
        class_ids = tf.argmax(rcnn_probs, axis=1, output_type=tf.int32)

        # class probability of the top class of each roi
        indices = tf.stack([tf.range(tf.shape(rcnn_deltas)[0]), class_ids], axis=1)
        class_scores = tf.gather_nd(rcnn_probs, indices)
        bounding_deltas = tf.gather_nd(rcnn_deltas, indices)

        # apply bounding box deltas
        refined_rois = deltas2bbox(rois, bounding_deltas, self.target_means, self.target_stds)
        # debug mode: as there are some nan values so put it 0 instead of nan
        refined_rois = tf.where(tf.math.is_nan(refined_rois), 0, refined_rois)
        # clipped rois out of windows
        refined_rois *= tf.constant([H, W, H, W], dtype=tf.float32)
        window = tf.constant([0.0, 0.0, 1.0 * H, 1.0 * W], dtype=tf.float32)

        wy1, wx1, wy2, wx2 = tf.split(window, 4)
        y1, x1, y2, x2 = tf.split(refined_rois, 4, axis=1)

        # clip
        y1 = tf.maximum(tf.minimum(wy1, y1), wy1)
        x1 = tf.maximum(tf.minimum(wx1, x1), wx1)
        y2 = tf.maximum(tf.minimum(wy2, y2), wy1)
        x2 = tf.maximum(tf.minimum(wx2, x2), wx1)

        clipped = tf.concat([y1, x1, y2, x2], axis=1)
        clipped.set_shape((clipped.shape[0], 4))

        # filter background anchors
        keep_indices = tf.where(class_ids > 0)[:, 0]

        # Filter out low confidence boxes
        if self.min_confidence:
            conf_keep = tf.where(class_scores >= self.min_confidence)[:, 0]
            keep_indices = tf.compat.v2.sets.intersection(tf.expand_dims(keep_indices, 0),
                                                          tf.expand_dims(conf_keep, 0))
            keep_indices = tf.sparse.to_dense(keep_indices)[0]

        # Apply per-class NMS
        # 1. Prepare variable
        pre_nms_class_ids = tf.gather(class_ids, keep_indices)
        pre_nms_scores = tf.gather(class_scores, keep_indices)
        pre_nms_rois = tf.gather(refined_rois, keep_indices)
        unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

        def nms_keep_map(class_id):
            indices = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
            # apply nms
            # returns selected_indices
            nms_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, indices),
                tf.gather(pre_nms_scores, indices),
                iou_threshold=self.nms_threshold,
                max_output_size=self.max_instances
            )

            class_id_keep = tf.gather(keep_indices, tf.gather(indices, nms_keep))
            return class_id_keep

        nms_keep = []
        for i in range(unique_pre_nms_class_ids.shape[0]):
            nms_keep.append(nms_keep_map(unique_pre_nms_class_ids[i]))

        nms_keep = tf.concat(nms_keep, axis=0)

        # 3. Compute intersection between keep and nms_keep
        # todo check why here so complicated
        keep = tf.compat.v2.sets.intersection(tf.expand_dims(keep_indices, 0),
                                              tf.expand_dims(nms_keep, 0))
        keep = tf.sparse.to_dense(keep)[0]

        # Keep top detections
        rois_count = self.max_instances
        class_keep_scores = tf.gather(class_scores, keep)
        num_keep = tf.minimum(class_keep_scores.shape[0], rois_count)
        top_ids = tf.nn.top_k(class_keep_scores, k=num_keep, sorted=True)[1]

        keep = tf.gather(keep, top_ids)

        detection = tf.concat([
            tf.gather(refined_rois, keep),  # [N, 4]
            tf.cast(tf.gather(class_ids, keep), tf.float32)[..., tf.newaxis],  # [N, 1]
            tf.gather(class_scores, keep)[..., tf.newaxis],  # [N ,1]
        ], axis=1)

        return detection
