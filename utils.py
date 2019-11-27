def parse_image_meta(meta):
    '''
    Parses a tensor that contains image attributes to its components.

    Args
    ---
        meta: [..., 11]

    Returns
    ---
        a dict of the parsed tensors.
    '''
    meta = meta.numpy()
    image_id = meta[..., 0]
    ori_shape = meta[..., 1:3]
    img_shape = meta[..., 4:6]
    pad_shape = meta[..., 7:9]
    # scale = meta[..., 9]
    # flip = meta[..., 10]
    return {
        'ori_shape': ori_shape,
        'img_shape': img_shape,
        'pad_shape': pad_shape,
        # 'scale': scale,
        # 'flip': flip
    }


def bbox_mapping_back(box, img_meta):
    '''
    Args
    ---
        box: [N, 4]
        img_meta: [11]
    '''
    img_meta = parse_image_meta(img_meta)
    scale = img_meta['scale']
    # flip = img_meta['flip']
    # if tf.equal(flip, 1):
    #     box = bbox_flip(box, img_meta['img_shape'][1])
    box = box / scale

    return box



