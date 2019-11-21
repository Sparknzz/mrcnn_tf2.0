import tensorflow as tf
from tensorflow.keras import layers
from mrcnn.loss import losses


class MaskHead(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(MaskHead, self).__init__(**kwargs)

        self.num_classes = num_classes

        # Conv layers
        self.mask_conv1 = layers.Conv2D(256, (3, 3), padding="same", name='mrcnn_mask_conv1')
        self.mask_bn1 = layers.BatchNormalization(name='mrcnn_mask_bn1')

        self.mask_conv2 = layers.Conv2D(256, (3, 3), padding="same", name='mrcnn_mask_conv2')
        self.mask_bn2 = layers.BatchNormalization(name='mrcnn_mask_bn2')

        self.mask_conv3 = layers.Conv2D(256, (3, 3), padding="same", name='mrcnn_mask_conv3')
        self.mask_bn3 = layers.BatchNormalization(name='mrcnn_mask_bn3')

        self.mask_conv4 = layers.Conv2D(256, (3, 3), padding="same", name='mrcnn_mask_conv4')
        self.mask_bn4 = layers.BatchNormalization(name='mrcnn_mask_bn4')

        self.mask_deconv = layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu",
                                                  name="mrcnn_mask_deconv")  # 14,14,256
        self.mask_probs = layers.Conv2D(self.num_classes, (1, 1), strides=1, activation="sigmoid", name="mrcnn_mask")

        # loss
        self.mask_loss = losses.rcnn_mask_loss

    def call(self, pooled_rois_list, training=True):
        '''
        pooled_rois : list of pooled proposals => [(num_rois, H, W, channel),(num_rois, H, W, channel),(num_rois, H, W, channel)]
        shape [batch, num_rois, H, W, channel]

        return:

        '''
        num_pooled_rois_list = [pooled_rois.shape[0] for pooled_rois in pooled_rois_list]
        pooled_rois = tf.concat(num_pooled_rois_list, axis=0)

        x = self.mask_conv1(pooled_rois)
        x = self.mask_bn1(x)
        x = tf.nn.relu(x)

        x = self.mask_conv2(x)
        x = self.mask_bn2(x)
        x = tf.nn.relu(x)

        x = self.mask_conv3(x)
        x = self.mask_bn3(x)
        x = tf.nn.relu(x)

        x = self.mask_conv4(x)
        x = self.mask_bn4(x)
        x = tf.nn.relu(x)

        x = self.mask_deconv(x)
        probs = self.mask_probs(x)  # [n, num_class*2, H, W]

        # note here probs including all batches mask so split them
        mrcnn_mask_list = tf.split(probs, num_pooled_rois_list, axis=0)

        return mrcnn_mask_list

    def loss(self, target_mask_list, target_class_ids_list, mrcnn_mask_list):
        '''
            target_mask_list: list of target mask
            target_class_ids: list of target class ids
            mrcnn_mask: list of pred mrcnn masks

        '''
        loss = self.mask_loss(target_mask_list, target_class_ids_list, mrcnn_mask_list)
        return loss
