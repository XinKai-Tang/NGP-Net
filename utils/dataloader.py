import os
import json
import numpy as np

from argparse import ArgumentParser as Parser
from torch.utils.data import DataLoader, Dataset

from . import transforms as T


class DatasetNG3T(Dataset):
    ''' Nodule Growth Dataset with 3 Time Points '''

    def __init__(self, 
                 data_dir: str,
                 data_list: list,
                 transforms: object):
        ''' Args:
        * `data_dir`: save path of dataset.
        * `data_list`: filenames of data in this fold.
        * `transforms`: transform functions for dataset.
        '''
        super(DatasetNG3T, self).__init__()
        self.data_dir = data_dir
        self.data_list = data_list
        self.transforms = transforms

    def __getitem__(self, index: int):
        ''' get images, masks, intervals and infomation by index '''
        info = self.data_list[index]["info"]
        itvs = [int(info["Months1"]), int(info["Months2"])]
        images, labels = [], []
        for series in self.data_list[index]["series"]:
            images.append(os.path.join(self.data_dir, series["image"]))
            labels.append(os.path.join(self.data_dir, series["label"]))
        images, labels = self.transforms(images, labels)
        return *images, *labels, *itvs, info
    
    def __len__(self):
        ''' get length of the dataset '''
        return len(self.data_list)
    

def get_transforms(args: Parser, is_test: bool = False):
    SCOPE = (args.s_min, args.s_max)    # scope of CT voxels
    RANGE = (args.r_min, args.r_max)    # range of model inputs
    SHAPE = [args.in_size] * 3          # shape of model inputs

    if not is_test:
        tra_transfoms = T.Compose([
            T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
            T.RandomCrop(rand_crop=False, crop_size=SHAPE),
            T.ScaleIntensity(scope=SCOPE, range=RANGE),
            T.RandomFilp(prob=0.2, axes=(0, 1, 2)),
            T.RandomRot90(prob=0.2, axes=(0, 1, 2)),
            T.RandomScaleIntensity(prob=0.1, factor=0.1),
            T.RandomShiftIntensity(prob=0.1, offset=0.1*(RANGE[1]-RANGE[0])),
            T.AddChannel(img_add=True, msk_add=True),
            T.ToTensor()
        ])
        val_transforms = T.Compose([
            T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
            T.RandomCrop(rand_crop=False, crop_size=SHAPE),
            T.ScaleIntensity(scope=SCOPE, range=RANGE),
            T.AddChannel(img_add=True, msk_add=True),
            T.ToTensor()
        ])
        return (tra_transfoms, val_transforms)
    else:
        test_transforms = T.Compose([
            T.LoadImage(img_dtype=np.float32, msk_dtype=np.uint8),
            T.RandomCrop(rand_crop=False, crop_size=SHAPE),
            T.ScaleIntensity(scope=SCOPE, range=RANGE),
            T.AddChannel(img_add=True, msk_add=True),
            T.ToTensor()
        ])
        return test_transforms


def get_loader(args: Parser, is_test: bool = False):
    transforms = get_transforms(args, is_test)

    if not is_test:
        data_list = get_data_list(args, "training")
        tra_data, val_data = split_data_list(
            data_list, args.num_folds, fold=args.fold
        )
        train_dataset = DatasetNG3T(data_dir=args.data_root,
                                    data_list=tra_data,
                                    transforms=transforms[0])
        train_loader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

        val_dataset = DatasetNG3T(data_dir=args.data_root,
                                  data_list=val_data,
                                  transforms=transforms[1])
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=args.num_workers)
        return train_loader, val_loader
    else:
        test_data = get_data_list(args, "test")
        test_dataset = DatasetNG3T(data_dir=args.data_root,
                                   data_list=test_data,
                                   transforms=transforms)
        test_loader = DataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=False,
                                 num_workers=args.num_workers)
        return test_loader


def get_data_list(args: Parser, key: str):
    json_path = os.path.join(args.data_root, args.json_name)
    with open(json_path, "r") as jsf:
        json_data = json.load(jsf)[key]
    assert isinstance(json_data, list) and len(json_data) > 0
    return json_data


def split_data_list(data_list: list, num_folds: int, fold: int = 0):
    assert fold < num_folds, "`fold` should be less than `num_folds`."
    tra_data, val_data = [], []
    for idx, item in enumerate(data_list):
        if idx % num_folds == fold:
            val_data.append(item)
        else:
            tra_data.append(item)
    return tra_data, val_data