from datasets import my_dataset
from datasets.data_generator import *
import config

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.__version__)
assert tf.__version__.startswith('2.')
tf.random.set_seed(22)
np.random.seed(22)

img_mean = (123.675, 116.28, 103.53)
# img_std = (58.395, 57.12, 57.375)
img_std = (1., 1., 1.)

batch_size = 2
num_classes = 2

train_dataset = my_dataset.MyDataSet()
train_dataset.load_balloon('data', 'train')
train_dataset.prepare()

# create data generator
train_generator = data_generator(train_dataset, config=config.Config(), shuffle=True,
                                 augmentation=None,
                                 batch_size=batch_size)

batch_images, batch_image_meta, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks = next(train_generator)

from visualize import *

from mrcnn.anchor import anchor_generator

generator = anchor_generator.AnchorGenerator()

anchors, valid_flags = generator.generate_pyramid_anchors(batch_image_meta)

# anchors_list = list(anchors.numpy())[258870:258875]
gt = list(batch_gt_boxes[1])


def compute_overlaps(anchors, gt_boxes):
    '''
        anchors: [N,4]
        gt_boxes: [N,4]

        return: [anchors, gts]
    '''
    # tile anchors and repeat gt_boxes
    b1 = tf.reshape(tf.tile(tf.expand_dims(anchors, 1), [1, 1, tf.shape(gt_boxes)[0]]), [-1, 4])
    b2 = tf.tile(gt_boxes, [tf.shape(anchors)[0], 1])

    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, -1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, -1)

    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)

    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)

    union = b1_area + b2_area - intersection

    iou = intersection / union  # eg anchor 20 gt 10, iou: [200,1]  reshape to [20, 10]

    overlaps = tf.reshape(iou, [tf.shape(anchors)[0], tf.shape(gt_boxes)[0]])

    return overlaps


# overlaps = compute_overlaps(anchors, tf.cast(gt, dtype=tf.float32))

# idx = tf.where(tf.greater(overlaps, 0.7))

# print(idx)
# all_anchor = np.concatenate([anchors_list, gt], axis=0)

# draw_boxes(batch_images[0], all_anchor)


from mrcnn import mask_rcnn

model = mask_rcnn.MaskRCNN(2, config.Config())

proposal_list = model([batch_images, batch_image_meta, batch_gt_class_ids, batch_gt_boxes, batch_gt_masks])

proposal0 = proposal_list[1].numpy()

H = 1024
W = 1024
proposal0 *= np.array([H, W, H, W], dtype=np.int32)

gt = list(batch_gt_boxes[1])
overlaps = compute_overlaps(proposal0, tf.cast(gt, dtype=tf.float32))

idx = tf.where(tf.greater(overlaps, 0.5))

print(idx)

# for debugging
if True:
    draw_boxes(batch_images[0], proposal0)
