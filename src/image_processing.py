from toolz import curry
from skimage.transform import resize


@curry
def upsample(original, target, img):
    if original == target:
        return img
    return resize(img, (original, target), mode='constant', preserve_range=True, anti_aliasing=False)

@curry
def downsample(original, target, img):
    if original == target:
        return img
    return resize(img, (original, target), mode='constant', preserve_range=True, anti_aliasing=False)