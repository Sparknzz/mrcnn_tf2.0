import tensorflow as tf


class PyramidROIAlign(tf.keras.layers.Layer):
    def __init__(self, pool_shape, **kwargs):
        '''
        Implements ROI Pooling on multiple levels of the feature pyramid.

        Attributes
        ---
            pool_shape: (height, width) of the output pooled regions.
                Example: (7, 7)
        '''
        super().__init__(kwargs)
        self.pool_shape = pool_shape

    def call(self, inputs):
        '''
        :param inputs:
        rois_list = [tf.Variable(proposals)],
        feature maps list: list of feature maps, [(Batch, H, W, C),(Batch, H, W, C)....] P2-P5
        img_metas.
        :return: list of pooled proposals

        NOTE: For simple test, there is one batch, so no batch dim, but multiple batch it should be [batch,N,4]
        NOTE: rois_list is proposals which is [batch, 2000, 4]
        '''
        # [Tensor(2000,(y1,x1,y2,x2)),...] normalized coordinates feature maps [C2,C3,C4,C5,C6]
        rois_list, feature_maps_list, img_metas = inputs

        pad_shapes = tf.cast(img_metas[:, 7:9], dtype=tf.float32).numpy()

        pad_areas = pad_shapes[:, 0] * pad_shapes[:, 1]  # [batch, h*w]

        # find out rois belongs to which number of the batch
        roi_indices = tf.constant([
            i for i in range(pad_areas.shape[0]) for _ in range(rois_list[i].shape.as_list()[0])],
            dtype=tf.int32
        )  # [0,0,0,0,0...,1,1,1,1,1,2,2,2,2....] [shape 2000 * batch] 2000 times 0, 2000 times 1, 2000 times2

        # image area
        num_rois_list = [rois.shape.as_list()[0] for rois in rois_list]  # data:[2000, 2000, ... ] how many batches

        areas = tf.constant([pad_areas[i] for i in range(pad_areas.shape[0]) for _ in range(num_rois_list[i])],
                            dtype=tf.float32)

        rois = tf.concat(rois_list, axis=0)

        # 1. assign each roi to a level in pyramid based on ROI area
        y1, x1, y2, x2 = tf.split(rois, 4, axis=-1)
        h = y2 - y1
        w = x2 - x1

        # 1. equation to calculate proposals belongs to which feature map
        # Equation 1 in the Feature Pyramid Networks paper. Account for
        # the fact that our coordinates are normalized here.
        # e.g. a 224x224 ROI (in pixels) maps to P4
        roi_level = tf.math.log(  # [2000]
            tf.sqrt(tf.squeeze(h * w, 1))
            / tf.cast((224.0 / tf.sqrt(areas * 1.0)), tf.float32)) / tf.math.log(2.0)

        roi_level = tf.minimum(5, tf.maximum(  # [2000], clamp to [2-5]
            2, 4 + tf.cast(tf.round(roi_level), tf.int32)))

        # enumerate all pyramid levels and collect corresponding roi pooling
        pooled_rois = []
        roi_to_level = []

        for i, level in enumerate(range(2, 6)):
            # 1. find indx of rois_level
            ix = tf.where(tf.equal(roi_level, level))  # roi_level: [2000]
            level_rois = tf.gather_nd(rois, ix)  # rois:[2000,4] ix:[N,1] tf where return n,1 shape

            # ROI indices for crop and resize
            level_roi_indices = tf.gather_nd(roi_indices, ix)

            # Keep track of which roi is mapped to which level
            roi_to_level.append(ix)

            # stop gradients
            level_rois = tf.stop_gradient(level_rois)
            level_roi_indices = tf.stop_gradient(level_roi_indices)

            # Crop and Resize
            # From Mask R-CNN paper: "We sample four regular locations, so
            # that we can evaluate either max or average pooling. In fact,
            # interpolating only a single value at each bin center (without
            # pooling) is nearly as effective."
            #
            # Here we use the simplified approach of a single value per bin,
            # which is how it's done in tf.crop_and_resize()
            # Result: [batch * num_rois, pool_height, pool_width, channels]
            pooled_rois.append(tf.image.crop_and_resize(
                feature_maps_list[i], level_rois, level_roi_indices, self.pool_shape,
                method='bilinear'))

        # now we already got each level's pooled_rois, however, it is not corresponding to original rois_list
        # so next we need to sort pooled_rois to corresponding to which image it from
        pooled_rois = tf.concat(pooled_rois, axis=0)

        # concat roi in one array
        roi_to_level = tf.concat(roi_to_level, axis=0)
        roi_range = tf.expand_dims(tf.range(tf.shape(roi_to_level)[0]),
                                   1)  # eg. batch is 3 => [2000*3, 1] => [2000*3, 1]
        roi_to_level = tf.concat([tf.cast(roi_to_level, tf.int32), roi_range], axis=1)  # [2000, 2], (P, range)

        # Sort roi_to_level by batch then roi index
        # cos roi_to_level stores every level's roi, so we need to resort it to match original rois
        # as original rois is list of [batch, number rois, 4]
        # NOTE: eg batch 3 10 rois each, 30 rois in total, then 30 rois belongs to different level,
        # we know the level they belongs, and the indx which is roi_to_level. eg [1, 3, 4, 10, 20, 30.......]
        # we have to find the corresponding original ix, eg no 2 belongs to level 5, which is the last one,
        #
        sorting_tensor = roi_to_level[:, 0] * 100000 + roi_to_level[:, 1]
        ix = tf.nn.top_k(sorting_tensor, k=tf.shape(roi_to_level)[0]).indices[::-1]  # reverse the order
        ix = tf.gather(roi_to_level[:, 1], ix)
        pooled_rois = tf.gather(pooled_rois, ix)
        # 2000 of [7, 7, 256]
        pooled_rois_list = tf.split(pooled_rois, num_rois_list, axis=0)
        return pooled_rois_list
