import numpy as np
import cv2


class ImageTransform(object):
    '''Preprocess the image.
        1. rescale the image to expected size
        2. normalize the image
        3. flip the image (if needed)
        4. pad the image (if needed)
    '''

    def __init__(self, scale=(800, 1024),
                 mean=(0, 0, 0),
                 std=(1, 1, 1),
                 pad_mode='fixed'):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.pad_mode = pad_mode

        self.impad_size = max(scale) if pad_mode == 'fixed' else 64

    def __call__(self, img, flip=False):
        img, scale_factor = im_rescale(img, self.scale)
        scaled_img_shape = img.shape
        img = im_normalize(img, self.mean, self.std)

        # if flip:
        # img = img_flip(img)
        if self.pad_mode == 'fixed':
            img = impad_to_square(img, self.impad_size)

        # else:  # 'non-fixed'
        #     img = impad_to_multiple(img, self.impad_size)

        return img, scaled_img_shape, scale_factor


def im_normalize(img, mean, std):
    img = (img - mean) / std
    return img.astype(np.float32)


def im_rescale(img, scale):
    '''Resize image while keeping the aspect ratio.

    Args
    ---
        img: [height, width, channels]. The input image.
        scale: Tuple of 2 integers. the image will be rescaled
            as large as possible within the scale

    Returns
    ---
        np.ndarray: the scaled image.
    '''
    h, w = img.shape[:2]

    max_long_edge = max(scale)
    max_short_edge = min(scale)
    scale_factor = min(max_long_edge / max(h, w),
                       max_short_edge / min(h, w))

    new_size = (int(w * float(scale_factor) + 0.5),
                int(h * float(scale_factor) + 0.5))

    rescaled_img = cv2.resize(
        img, new_size, interpolation=cv2.INTER_LINEAR)

    return rescaled_img, scale_factor


def impad_to_square(img, pad_size):
    shape = (pad_size, pad_size, img.shape[-1])

    pad = np.zeros(shape, dtype=img.dtype)
    pad[:img.shape[0], :img.shape[1], ...] = img
    return pad
