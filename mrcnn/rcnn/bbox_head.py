import tensorflow as tf
from tensorflow.keras import layers
from mrcnn.loss import losses


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
