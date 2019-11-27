import json
import os

from datasets.custom_transform import *
from datasets.dataset import *


class MyDataSet(Dataset):

    def load_balloon(self, dataset_dir, subset):
        """Load a subset of the Balloon dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        source = 'balloon'
        # Add classes. We have only one class to add.
        self.add_class(source, 1, "balloon")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        image_dir = os.path.join(dataset_dir, subset, 'images')

        anno_dir = os.path.join(dataset_dir, subset, 'annotations')

        # normally we will have our own dataset,
        # it apparently should be a listdir, or sometimes can be .npy file for split folds
        image_list = os.listdir(image_dir)
        image_names = []  # normally ids is unique id == name

        for img in image_list:
            image_names.append(img[:-4])

        # Load annotations
        # {
        #   'version': '3.16.1',
        #   'flags': {},
        #   'shapes' : [
        #       {
        #           'label':'xx'
        #           'line_color':null
        #           'fill_color':null
        #           'points':[[x,y],[x,y],[x,y]]
        #       },
        #       {},
        #       {},
        #       {},
        #   ],
        #   'lineColor':
        #   'fillColor':
        #   'imagePath':
        #   'imageData': null
        #   'imageHeight': 1000
        #   'imageWidth': 1000
        # }

        for i in range(len(image_names)):
            annotations = json.load(open(os.path.join(anno_dir, image_names[i] + ".json")))

            shapes = annotations['shapes']  # shapes include all region

            polygons = [shape['points'] for shape in shapes]  # list of list

            height = annotations['imageHeight']
            width = annotations['imageWidth']

            self.add_image(
                source=source,
                image_id=i,
                path=os.path.join(dataset_dir, subset, 'images', image_names[i] + '.jpg'),
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        index for id in the list, not image_id.
        images_info is a [{'source':, 'class_id':, 'path':}]

        in this stage, depends on the annotation choice, you can load json file or mask png file
        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        image_info = self.images_info[image_id]

        # root_dir = ''
        # Get mask directory from image path normally our dir is SegmentationClassPNG
        # mask_dir = os.path.join(root_dir, 'SegmentationClassPNG')

        mask = np.zeros([image_info["height"], image_info["width"], len(image_info["polygons"])],
                        dtype=np.uint8)  # eg 1000*1000*10 mask
        for i, p in enumerate(image_info["polygons"]):
            p = np.array(p).astype(np.int32)
            all_points_x = p[:, 0]
            all_points_y = p[:, 1]
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones
        return mask, np.ones([mask.shape[-1]], dtype=np.int32)


def compose_image_meta(image_id, original_image_shape, scaled_img_shape,
                       pad_shape, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing
    pad_shape: [H, W, C] after padding
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on data from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +  # size=1
        list(original_image_shape) +  # size=3
        list(scaled_img_shape) +  # size=3
        list(pad_shape) +  # size=3
        [scale[0]] +  # size=1 NO LONGER, I dont have time to correct this properly so take only the first element
        list(active_class_ids)  # size=num_classes
    )

    return meta
