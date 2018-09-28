import numpy as np
from tqdm import tqdm
import os
from toolz import compose
import pandas as pd
from PIL import Image
from metrics import cov_to_class
from torch.utils import data


def training_images_path():
    return os.path.join(os.getenv('DATA'), 'train', "images")


def training_masks_path():
    return os.path.join(os.getenv('DATA'), 'train', "masks")

def test_images_path():
    return os.path.join(os.getenv('DATA'), 'test', "images")

def training_csv_path():
    return os.path.join(os.getenv('DATA'), "train.csv")


def depths_csv_path():
    return os.path.join(os.getenv('DATA'), "depths.csv")


def _load_and_normalize(path):
    image_grey = Image.open(path).convert('L')
    return (np.array(image_grey) / 255)#.reshape(image_grey.height, image_grey.width, 1)


def load_images_as_arrays(image_id_iter, images_path, progress=tqdm):
    process = compose(_load_and_normalize,
                      lambda idx: os.path.join(images_path, "{}.png".format(idx)))
    return list(map(process, progress(image_id_iter)))


def _read_train_depth_csv():
    train_df = pd.read_csv(training_csv_path(), index_col="id", usecols=[0])
    depths_df = pd.read_csv(depths_csv_path(), index_col="id")
    return train_df, depths_df


def prepare_training_data(img_size_ori=101):
    train_df, depths_df = _read_train_depth_csv()
    train_df = train_df.join(depths_df)
   
    train_df["images"] = load_images_as_arrays(train_df.index, training_images_path())
    train_df["masks"] = load_images_as_arrays(train_df.index, training_masks_path())

    train_df["coverage"] = train_df.masks.map(np.sum) / pow(img_size_ori, 2)
    train_df["coverage_class"] = train_df.coverage.map(cov_to_class)
    return train_df


def prepare_test_data():
    train_df, depths_df = _read_train_depth_csv()
    test_df = depths_df[~depths_df.index.isin(train_df.index)]
    return test_df


def prepare_data(img_size_ori=101):
    train_df = prepare_training_data(img_size_ori=img_size_ori)
    test_df = prepare_test_data()
    return train_df, test_df



def run_length_encode(img, order='F', format=True):
    """ Run length encoding of image
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    # Source https://www.kaggle.com/bguberfain/unet-with-depth
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    # if last run is unsaved (i.e. data ends with 1)
    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs
    
    
"""
used for converting the decoded image to rle mask
Fast compared to previous one
"""
def rle_encode(im):
    '''
    im: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    pixels = im.flatten(order = 'F')
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


class TGSSaltDataset(data.Dataset):
    def __init__(self, x_data, y_data=None, is_test=False):
        self._is_test = is_test
        self._x_data = x_data
        self._y_data = y_data

    @property
    def is_test(self):
        return self._is_test

    def __len__(self):
        return len(self._x_data)

    def __getitem__(self, index):
        image = self._x_data[index]

        if self.is_test:
            return image
        else:
            mask = self._y_data[index]
            return image, mask