import tensorflow as tf


class AnchorGenerator():
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

    def generate_pyramid_anchors(self, img_metas):
        '''
        Generate the multi-level anchors for Region Proposal Network

        Args
        ---
            img_metas: [batch_size, 11]

        Returns
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
        '''

        # generate anchors

        pass
