import numpy as np

# taken from https://github.com/xkumiyu/numpy-data-augmentation/blob/master/process_image.py
def random_crop(image, crop_size):
    crop_size = check_size(crop_size)
    h, w, _ = image.shape
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])
    bottom = top + crop_size[0]
    right = left + crop_size[1]
    image = image[top:bottom, left:right, :]
    return image

# adapted from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/image_ops_impl.py
def central_crop(image, central_fraction):
    H, W, C = image.shape
    # gap to leave on one size H*central_fraction = H' so (H - H') / 2 = gap on one side
    h_start = int((H - H * central_fraction) / 2)
    # same as above
    w_start = int((W - W * central_fraction) / 2)
    # H - 2*(one_side_gap) = H'
    h_size  = H - h_start * 2
    # same as above
    w_size  = W - w_start * 2
    # perform crop
    new_image = image[h_start:h_start+h_size, w_start:w_start+w_size, :]
    return new_image

# taken from https://github.com/xkumiyu/numpy-data-augmentation/blob/master/process_image.py
def horizontal_flip(image, rate=0.5):
    if np.random.rand() < rate:
        image = image[:, ::-1, :]
    return image
