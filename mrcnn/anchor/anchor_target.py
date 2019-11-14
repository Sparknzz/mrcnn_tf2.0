from mrcnn.utils import *

"""
for every generated anchors boxes:
create its rpn_target_matchs and rpn_target_deltas
NOTE : VERY IMPORTANT you might think when do the regression, it should be proposals and gts.
but the thing is only anchors will do all calculate thing. this class is used for calculate deltas for all anchors and gts.
only returns 256 deltas for rpn training
"""

class AnchorTarget:
    def __init__(self,
                 target_means=(0., 0., 0., 0.),
                 target_stds=(0.1, 0.1, 0.2, 0.2),
                 num_rpn_deltas=256,
                 positive_fraction=0.5,
                 pos_iou_thr=0.7,
                 neg_iou_thr=0.3):
        '''
        Compute regression and classification targets for anchors.

        Attributes
        ---
            target_means: [4]. Bounding box refinement mean for RPN.
            target_stds: [4]. Bounding box refinement standard deviation for RPN.
            num_rpn_deltas: int. Maximal number of Anchors per image to feed to rpn heads.
            positive_fraction: float.
            pos_iou_thr: float.
            neg_iou_thr: float.
        '''

        self.target_means = target_means
        self.target_stds = target_stds
        self.num_rpn_deltas = num_rpn_deltas
        self.positive_fraction = positive_fraction
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr

    def build_targets(self, anchors, gt_boxes, gt_class_ids):
        '''
        Given the anchors and GT boxes, compute overlaps and identify positive
        anchors and deltas to refine them to match their corresponding GT boxes.

        Args
        ---
            anchors: [num_anchors, (y1, x1, y2, x2)] in image coordinates.
            valid_flags: [batch_size, num_anchors]
            gt_boxes: [batch_size, num_gt_boxes, (y1, x1, y2, x2)] in image
                coordinates. batch_size = 1 usually
            gt_class_ids: [batch_size, num_gt_boxes] Integer class IDs.

        Returns
        ---
            rpn_target_matchs: [batch_size, num_anchors] matches between anchors and GT boxes.
                1 = positive anchor, -1 = negative anchor, 0 = neutral anchor
            rpn_target_deltas: [batch_size, num_rpn_deltas, (dy, dx, log(dh), log(dw))]
                Anchor bbox deltas.
        '''
        rpn_target_matches = []
        rpn_target_deltas = []

        num_images = gt_boxes.shape[0]  # namely, batch , 1

        for i in range(num_images):
            self._build_single_img_target(anchors, gt_boxes[i], gt_class_ids[i])
        pass
