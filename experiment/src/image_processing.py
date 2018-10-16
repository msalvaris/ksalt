from skimage.transform import resize
from toolz import curry


@curry
def upsample(original, target, img):
    if original == target:
        return img
    return resize(img, (target, target), mode='constant', preserve_range=True, anti_aliasing=False)

@curry
def downsample(original, target, img):
    if original == target:
        return img
    return resize(img, (target, target), mode='constant', preserve_range=True, anti_aliasing=False)