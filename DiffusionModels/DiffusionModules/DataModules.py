from lightning.pytorch import LightningDataModule
from DiffusionModules.Util import ImageTransformer
from abc import ABC, abstractmethod
import torch
import torchvision
import braceexpand
import operator
import functools
from enum import Enum
import webdataset as wds

class CollateTypeFunction():
    @staticmethod
    def cnd(data):
        data = list(filter(lambda x: x[0] is not None and x[1] is not None, data))
        images = [e[0] for e in data]
        captions = [e[1] for e in data]
        return {"image": images, "caption": captions}

    @staticmethod
    def cnt(data):
        data = list(filter(lambda x: x[0] is not None and x[1] is not None, data))
        images = [e[0] for e in data]
        captions = [e[1] for e in data]
        return images, captions

    @staticmethod
    def ci(data):
        return data

collate_type_to_function = {
    "COLLATE_NONE_TUPLE": CollateTypeFunction.cnt,
    "COLLATE_NONE_DICT": CollateTypeFunction.cnd,
    "COLLATE_IDENTITY": CollateTypeFunction.ci
}

class CollateType(Enum):
    COLLATE_NONE_TUPLE="COLLATE_NONE_TUPLE"
    COLLATE_NONE_DICT="COLLATE_NONE_DICT"
    COLLATE_IDENTITY="COLLATE_IDENTITY"

    def __call__(self, *args):
        return collate_type_to_function[self.value](*args)


class TransformableImageDataModule(LightningDataModule, ABC):
    def __init__(self, collate_type=CollateType.COLLATE_NONE_TUPLE, collate_fn=None, img_in_target_size=64):
        super(TransformableImageDataModule, self).__init__()
        self.img_in_target_size = img_in_target_size
        self.transform = ImageTransformer(img_target_size=self.img_in_target_size)
        self.collate = collate_type if isinstance(collate_type, CollateType) else collate_fn

    @abstractmethod
    def train_dataloader(self):
        pass
    
    @abstractmethod
    def val_dataloader(self):
        pass

    def transform_images(self, batch):
        return self.transform.transform_images(batch)

    def reverse_transform_images(self, batch):
        return self.transform.reverse_transform_images(batch)

    def transform_image(self, image):
        return self.transform.transform_image(image)

    def reverse_transform_image(self, image):
        return self.transform.reverse_transform_image(image)


class CIFAR10DataModule(TransformableImageDataModule):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, cifar_path="/home/archive/cifar10-64", batch_size=16, num_workers=4, img_in_target_size=64):
        super(CIFAR10DataModule, self).__init__(CollateType.COLLATE_NONE_TUPLE, None, img_in_target_size)
        self.batch_size = batch_size
        self.cifar_path = cifar_path
        self.num_workers = num_workers
        self.collate_base = self.collate
        self.collate = lambda x: CIFAR10DataModule.collate_class_labels(self.collate_base(x))
    
    @staticmethod
    def collate_class_labels(x):
        return x[0], [CIFAR10DataModule.classes[e] for e in x[1]]

    def train_dataloader(self):
        trainset = torchvision.datasets.ImageFolder(f"{self.cifar_path}/train")
        return torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)

    def val_dataloader(self):
        valset = torchvision.datasets.ImageFolder(f"{self.cifar_path}/test")
        return torch.utils.data.DataLoader(valset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)


class WebdatasetDataModule(TransformableImageDataModule):
    @staticmethod
    def standard_preprocess(sample):
        image, json = sample
        label = json["caption"]
        return image, label

    def __init__(self, train_paths, val_paths, collate_type=CollateType.COLLATE_NONE_TUPLE, collate_fn=None, batch_size=16, num_workers=4, img_in_target_size=64):
        super(WebdatasetDataModule, self).__init__(collate_type, collate_fn, img_in_target_size)
        self.batch_size = batch_size
        self.train_paths = self.braceexpand_paths(train_paths)
        self.val_paths = self.braceexpand_paths(val_paths)
        self.num_workers = num_workers
        self.img_in_target_size = img_in_target_size
        self.transform = ImageTransformer(img_target_size=self.img_in_target_size)

    def braceexpand_paths(self, paths):
        if len(paths) == 1:
            return paths[0]

        return functools.reduce(operator.add, map(lambda a: list(braceexpand.braceexpand(a)), paths))

    def train_dataloader(self):
        ds = wds.WebDataset(self.train_paths, shardshuffle=True).shuffle(1000).decode("pil").to_tuple("jpg", "json").map(WebdatasetDataModule.standard_preprocess)
        return wds.WebLoader(ds, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate)

    def val_dataloader(self):
        ds = wds.WebDataset(self.val_paths, shardshuffle=True).shuffle(1000, initial=10000).decode("pil").to_tuple("jpg", "json").map(WebdatasetDataModule.standard_preprocess)
        return wds.WebLoader(ds, num_workers=self.num_workers, batch_size=self.batch_size, collate_fn=self.collate)

        
    