import os

from datasets.custom_transform import *


class DemoDataSet(object):
    def __init__(self, dataset_dir, mode, debug=False):
        '''Load a subset of the COCO dataset.

        Attributes
        ---
            dataset_dir: The root directory of the COCO dataset.
            mode: What to load (train, val).
            scale: Tuple of two integers.
        '''
        self.dataset_dir = dataset_dir
        self.image_info = []
        self.class_info = [{'id': 0, 'name': 'BG'}]
        self.mode = mode
        if mode not in ['train', 'val']:
            raise AssertionError('mode must be "train" or "val".')

    def add_class(self, class_id, class_name):
        for info in self.class_info:
            if info["id"] == class_id:
                return

        # Add the class
        self.class_info.append({
            "id": class_id,
            "name": class_name,
        })

    def add_image(self, image_id, path):
        image_info = {
            "id": image_id,
            "path": path,
        }

        self.image_info.append(image_info)

    def load_image(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class(1, "concrete")  # id, name

        paths = os.listdir(self.dataset_dir)

        # Add images
        for i in range(len(paths)):
            self.add_image(image_id=i, path=paths[i])

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. Override this
        method to load instance masks and return them in the form of am
        array of binary masks of shape [height, width, instances].

        Returns:
            masks: A bool array of shape [height, width, instance count] with
                a binary mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.
        """
        # Override this function to load a mask from your dataset.
        # Otherwise, it returns an empty mask.
        mask = np.empty([0, 0, 0])
        class_ids = np.empty([0], np.int32)
        return mask, class_ids

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        # todo need to implement mask logic here

        '''Load the image and its bboxes for the given index.
        Args
            idx: the index of images.
        Returns
        ---
            tuple: A tuple containing the following items: image, bboxes, labels.
        '''

        img_info = self.image_info[idx]

        # load the image.
        img = cv2.imread(os.path.join(self.dataset_dir, img_info['path']), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ori_shape = img.shape

        if self.mode == 'train':
            tr = ImageTransform()
            img, scaled_img_shape, scale_factor = tr(img)

        else:
            scaled_img_shape = img.shape
        #
        # else:
        #     pass

        img_meta = compose_image_meta(
            image_id=img_info['id'],
            original_image_shape=ori_shape,
            scaled_img_shape=scaled_img_shape,
            pad_shape=img.shape, scale=[8, 16, 32], active_class_ids=[1])

        return img, img_meta

def compose_image_meta(image_id, original_image_shape, scaled_img_shape,
                       pad_shape, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing
    pad_shape: [H, W, C] after padding
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
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
