import tensorflow as tf


class AnchorGenerator(object):
    """
       This class operate on padded iamge, eg. [1216, 1216]
       and generate scales*ratios number of anchor boxes for each point in
       padded image, with stride = feature_strides
       number of anchor = (1216 // feature_stride)^2
       number of anchor boxes = number of anchor * (scales_len*ratio_len)
       """

    def __init__(self,
                 scales=(32, 64, 128, 256, 512),
                 ratios=(0.5, 1, 2),
                 feature_strides=(4, 8, 16, 32, 64)):
        '''
        Anchor Generator

        Attributes
        ---
            scales: 1D array of anchor sizes in pixels. eg 8, 16, 32, 64, 128
            ratios: 1D array of anchor ratios of width/height. 0.5, 1, 2
            feature_strides: Stride of the feature map relative to the image in pixels. 4, 8, 16, 32, 64
        '''

        self.scales = scales
        self.ratios = ratios
        self.feature_strides = feature_strides

    def generate_pyramid_anchors(self, img_meta):
        """Generate anchor at different levels of a feature pyramid. Each scale
        is associated with a level of the pyramid, but each ratio is used in
        all levels of the pyramid.
            image_meta: ori_image id , shape, transformed image shape 0, 1, 2, 3, 4, 5, 6
        Returns:
        anchor: [N, (y1, x1, y2, x2)]. All generated anchor in one array. Sorted
            with the same order of the given scales. So, anchor of scale[0] come
            first, then anchor of scale[1], and so on.
            in my opinion, the anchors is in the image eg 512, 512, all anchor is not normalized pixel value
            valid_flags, each data valid flags, eg 2 data, each have 10000 anchors, then the flags is [20000,1]
        """
        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        # todo change here to re-factory code
        pad_shape = tf.cast(tf.reduce_max(img_meta[:, 4:6], axis=0), dtype=tf.int32).numpy()

        # <class 'list'>: [(256, 256), (128, 128), (64, 64), (32, 32), (16, 16)]
        feature_shapes = [(pad_shape[0] // stride, pad_shape[1] // stride)
                          for stride in self.feature_strides]

        anchors = [
            self._generate_level_anchors(level, feature_shape)
            for level, feature_shape in enumerate(feature_shapes)
        ]

        anchors = tf.concat(anchors, axis=0)

        # todo cleanup now cause normally we can leave all anchors now, no need to remove extra part.
        #  redo this for further calculation improvement.

        # need to find valid anchors means remove padding area anchors
        # eg. padding size 512*512 but before padding img_size is 256*256
        window = tf.cast(img_meta[..., 7:11], tf.int32).numpy()

        # generate valid flags means without padding area
        valid_flags = [
            self._generate_valid_flags(anchors, window[i])
            for i in range(window.shape[0])
        ]

        valid_flags = tf.stack(valid_flags, axis=0)

        anchors = tf.stop_gradient(anchors)
        valid_flags = tf.stop_gradient(valid_flags)

        return anchors, valid_flags

    def _generate_level_anchors(self, level, feature_shape):
        scale = self.scales[level]
        ratios = self.ratios
        feature_stride = self.feature_strides[level]

        # Get all combinations of scales and ratios
        scales, ratios = tf.meshgrid([float(scale)], ratios)
        scales = tf.reshape(scales, [-1])  # [8, 8, 8]
        ratios = tf.reshape(ratios, [-1])  # [0.5, 1, 2]

        # Enumerate heights and widths from scales and ratios
        heights = scales / tf.sqrt(ratios)
        widths = scales * tf.sqrt(ratios)

        # Enumerate shifts in feature space, [0*feature_stride]
        # if 512 then the shifts should be [0, 1*64, 2*64...., 15*64]
        shifts_y = tf.multiply(tf.range(feature_shape[0]), feature_stride)
        shifts_x = tf.multiply(tf.range(feature_shape[1]), feature_stride)

        shifts_x, shifts_y = tf.cast(shifts_x, tf.float32), tf.cast(shifts_y, tf.float32)
        shifts_x, shifts_y = tf.meshgrid(shifts_x, shifts_y)

        # Enumerate combinations of shifts, widths, and heights
        box_widths, box_centers_x = tf.meshgrid(widths, shifts_x)
        box_heights, box_centers_y = tf.meshgrid(heights, shifts_y)

        # Reshape to get a list of (y, x) and a list of (h, w)
        box_centers = tf.reshape(tf.stack([box_centers_y, box_centers_x], axis=-1), (-1, 2))
        box_sizes = tf.reshape(tf.stack([box_heights, box_widths], axis=-1), (-1, 2))

        # Convert to corner coordinates (N, (y1, x1, y2, x2))
        boxes = tf.concat([box_centers - 0.5 * box_sizes,
                           box_centers + 0.5 * box_sizes], axis=1)

        return boxes

    def _generate_valid_flags(self, anchors, window):
        '''
        remove these anchor boxed on padded area
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            window: (y1,x1,y2,x2) position of padded image

        Returns
        ---
            valid_flags: [num_anchors]
        '''

        img_y1, img_x1, img_y2, img_x2 = window

        y_center = (anchors[:, 2] + anchors[:, 0]) / 2
        x_center = (anchors[:, 1] + anchors[:, 3]) / 2

        valid_flags = tf.ones(anchors.shape[0], dtype=tf.int32)
        zeros = tf.zeros(anchors.shape[0], dtype=tf.int32)

        # set boxes whose center is out of image area as invalid.
        valid_flags = tf.where((y_center < img_y2) & (y_center > img_y1), valid_flags, zeros)
        valid_flags = tf.where((x_center < img_x2) & (x_center > img_x1), valid_flags, zeros)

        return valid_flags
