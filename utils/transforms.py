import numpy as np
import random
import torch

from typing import Union, Tuple
from utils.img_utils import crop_image, load_as_array, normalize_ct


class Compose:
    ''' compose transform functions '''

    def __init__(self, transforms: Tuple = []):
        self.transforms = transforms

    def __call__(self, images, masks=None):
        num = len(images)
        if masks is None:
            masks = [None] * num
        else:
            assert len(masks) == num
        for func in self.transforms:
            if hasattr(func, "update"):
                func.update()   # update random numbers
            for i in range(num):
                images[i], masks[i] = func(images[i], masks[i])
        return images, masks


class AddChannel:
    ''' add a channel dimension '''

    def __init__(self, img_add: bool = True, msk_add: bool = True):
        self.flags = (img_add, msk_add)

    def __call__(self, image, mask=None):
        if self.flags[0]:
            image = image[None]
        if self.flags[1] and mask is not None:
            mask = mask[None]
        return image, mask


class LoadImage:
    ''' load CT image as `numpy.ndarray` '''

    def __init__(self, img_dtype=np.float32, msk_dtype=np.uint8):
        self.img_dtype = img_dtype
        self.msk_dtype = msk_dtype

    def __call__(self, image: str, mask: str = None):
        image = load_as_array(image, self.img_dtype)
        if mask is not None:
            mask = load_as_array(mask, self.msk_dtype)
        return image, mask


class RandomCrop:
    ''' crop 3d image randomly '''

    def __init__(self, rand_crop: bool = True, 
                 crop_size: Tuple = (64, 64, 64)):
        self.rand_crop = rand_crop
        self.crop_size = crop_size

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        if mask is None:
            image = crop_image(img=image, rand_crop=self.rand_crop,
                               new_size=self.crop_size)
        else:
            image, mask = crop_image(img=image, msk=mask,
                                     new_size=self.crop_size,
                                     rand_crop=self.rand_crop)
        return image, mask


class RandomFilp:
    ''' flip 3d image randomly '''

    def __init__(self, prob: float = 0.5, axes: Tuple = (0, 1, 2)):
        assert 0 <= prob <= 1
        self.prob = prob
        self.axes = axes
        self.rand_nums = [0] * 3

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        if 0 in self.axes and self.rand_nums[0] < self.prob:
            image = image[..., ::-1, :, :]
            if mask is not None:
                mask = mask[..., ::-1, :, :]
        if 1 in self.axes and self.rand_nums[1] < self.prob:
            image = image[..., :, ::-1, :]
            if mask is not None:
                mask = mask[..., :, ::-1, :]
        if 2 in self.axes and self.rand_nums[2] < self.prob:
            image = image[..., :, :, ::-1]
            if mask is not None:
                mask = mask[..., :, :, ::-1]
        return image, mask
    
    def update(self):
        ''' update random numbers '''
        self.rand_nums = [random.random() for _ in self.rand_nums]


class RandomRot90:
    ''' rotate 3d image randomly '''

    def __init__(self, prob: float = 0.5, axes: Tuple = (0, 1, 2)):
        assert 0 <= prob <= 1
        self.prob = prob
        self.axes = axes
        self.rand_nums = [0] * 3

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        n_axes = len(self.axes)
        axes = [ax-3 for ax in self.axes]
        for i in range(0, n_axes-1):
            for j in range(i+1, n_axes):
                if self.rand_nums[i+j-1] < self.prob:
                    image = np.rot90(image, axes=(axes[i], axes[j]))
                    if mask is not None:
                        mask = np.rot90(mask, axes=(axes[i], axes[j]))
        return image, mask

    def update(self):
        ''' update random numbers '''
        self.rand_nums = [random.random() for _ in self.rand_nums]


class RandomShiftIntensity:
    ''' shift intensity randomly with offset '''

    def __init__(self, offset: float = 0.1, prob: float = 0.5):
        assert 0 <= offset
        assert 0 <= prob <= 1
        self.range = (-offset, offset)
        self.prob = prob
        self.rand_offset = 0.0
        self.rand_nums = [0] * 1

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        if self.rand_nums[0] < self.prob:
            image = np.asarray(image + self.rand_offset, dtype=image.dtype)
        return image, mask
    
    def update(self):
        self.rand_offset = random.uniform(self.range[0], self.range[1])
        self.rand_nums = [random.random() for _ in self.rand_nums]


class RandomScaleIntensity:
    ''' scale the intensity randomly with factor '''

    def __init__(self, factor: float = 0.1, prob: float = 0.5):
        assert 0 <= factor
        assert 0 <= prob <= 1
        self.range = (-factor, factor)
        self.prob = prob
        self.rand_factor = 1.0
        self.rand_nums = [0] * 1

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        if self.rand_nums[0] < self.prob:
            image = np.asarray(image * self.rand_factor, dtype=image.dtype)
        return image, mask
    
    def update(self):
        self.rand_factor = 1.0 + random.uniform(self.range[0], self.range[1])
        self.rand_nums = [random.random() for _ in self.rand_nums]


class ScaleIntensity:
    ''' scale intensity of the CT image '''

    def __init__(self, scope: Tuple = (-1200, 600),
                 range: Tuple = (-1, 1), clip: bool = True):
        assert float(scope[0]) < float(scope[1])
        assert float(range[0]) < float(range[1])
        self.scope = scope
        self.range = range
        self.clip = clip

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        image = normalize_ct(src=image, scope=self.scope,
                             range=self.range, clip=self.clip)
        return image, mask


class ToTensor:
    ''' convert inputs into `torch.Tensor` '''

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        image = torch.from_numpy(image.copy())
        if mask is not None:
            mask = torch.from_numpy(mask.copy())
        return image, mask
