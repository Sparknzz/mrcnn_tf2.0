import imgaug
import numpy as np


def augment(augmentation, image, mask):
    # Augmentors that are safe to apply to masks
    # Some, such as Affine, have settings that make them unsafe, so always
    # test your augmentation on masks
    MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                       "Fliplr", "Flipud", "CropAndPad",
                       "Affine", "PiecewiseAffine"]

    def hook(images, augmenter, parents, default):
        """Determines which augmenters to apply to masks."""
        return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

    # Store shapes before augmentation to compare
    image_shape = image.shape
    mask_shape = mask.shape
    # Make augmenters deterministic to apply similarly to data and masks
    det = augmentation.to_deterministic()
    image = det.augment_image(image)
    # Change mask to np.uint8 because imgaug doesn't support np.bool
    mask = det.augment_image(mask.astype(np.uint8),
                             hooks=imgaug.HooksImages(activator=hook))
    # Verify that shapes didn't change
    assert image.shape == image_shape, "Augmentation shouldn't change image size"
    assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
    # Change mask back to bool
    mask = mask.astype(np.bool)

    return image, mask
