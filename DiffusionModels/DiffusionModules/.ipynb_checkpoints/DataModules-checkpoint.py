from lightning.pytorch import LightningDataModule
from DiffusionModules.Util import ImageTransformer
from abc import ABC, abstractmethod
import torch
import torchvision
import braceexpand
import operator
import functools
from enum import Enum
from WebvidReader.VideoDataset import VideoDataset
import webdataset as wds
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence
from einops import rearrange
from torch.utils.data import DataLoader
from collections import OrderedDict


class CollateTypeFunction():
    standard_collations = OrderedDict(((0, "data"), (1, "caption")))
    standard_collations_with_fps = OrderedDict(((0, "data"), (1, "caption"), (2, "fps")))

    @staticmethod
    def additional_collate_none(data, collations=None):
        if collations is None:
            collations = CollateTypeFunction.standard_collations

        def filter_none(item, collations):
            valid = True
            for key in collations.keys():
                if item[key] is None:
                    valid = False
                    break
            return valid
        
        zipped = list(zip(*data))

        filtered_data = list(filter(lambda x: filter_none(x, collations), zipped))
        ret = OrderedDict([(collations[key], [item[key] for item in filtered_data]) for key in collations.keys()])
        return ret 

    @staticmethod
    def cnd(data, collations=None):
        data = CollateTypeFunction.additional_collate_none(data, collations)
        return data

    @staticmethod
    def cnt(data, collations=None):
        data = CollateTypeFunction.additional_collate_none(data, collations)
        return tuple(data.values())
        
    @staticmethod
    def ci(data):
        return data

    @staticmethod
    def cntps(data, padding_value=-1, collations=None):
        if collations is None:
            collations = CollateTypeFunction.standard_collations_with_fps
        data, captions, fps = CollateTypeFunction.cnt(list(zip(*data)), collations)
        lengths = torch.tensor([len(e) for e in data])
        if len(data) > 0:
            video = pad_sequence(data, padding_value=padding_value, batch_first=True)
        # video = pack_padded_sequence(video, lengths, batch_first=True)
        return video, captions, lengths, fps


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

class TransformableDataModule(LightningDataModule, ABC):
    @abstractmethod
    def train_dataloader(self):
        pass
    
    @abstractmethod
    def val_dataloader(self):
        pass
    
    @abstractmethod
    def test_dataloader(self):
        pass

    @abstractmethod
    def transform_batch(self, batch):
        pass

    @abstractmethod
    def reverse_transform_batch(self, batch):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def reverse_transform(self, data):
        pass


class TransformableImageDataModule(TransformableDataModule, ABC):
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
    
    @abstractmethod
    def test_dataloader(self):
        pass

    def transform_batch(self, batch):
        return self.transform.transform_images(batch)

    def reverse_transform_batch(self, batch):
        return self.transform.reverse_transform_images(batch)

    def transform(self, data):
        return self.transform.transform_image(data)

    def reverse_transform(self, data):
        return self.transform.reverse_transform_image(data)


class CIFAR10DataModule(TransformableImageDataModule):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, cifar_path="/home/archive/cifar10-64", batch_size=16, separate_val_split=False, separate_val_split_ratio=0.1, num_workers=4, img_in_target_size=64):
        super(CIFAR10DataModule, self).__init__(CollateType.COLLATE_NONE_TUPLE, None, img_in_target_size)
        self.batch_size = batch_size
        self.cifar_path = cifar_path
        self.num_workers = num_workers
        self.separate_val_split = separate_val_split
        self.separate_val_split_ratio = separate_val_split_ratio
        self.collate_base = self.collate
        self.collate = lambda x: CIFAR10DataModule.collate_class_labels(self.collate_base(x))
        
        if self.separate_val_split:
            trainset = torchvision.datasets.ImageFolder(f"{self.cifar_path}/train")
            self.val_set, self.train_set = torch.utils.data.random_split(trainset, [self.separate_val_split_ratio, 1.0 - self.separate_val_split_ratio])
            self.test_set = torchvision.datasets.ImageFolder(f"{self.cifar_path}/test")
        else:
            self.train_set = torchvision.datasets.ImageFolder(f"{self.cifar_path}/train")
            self.val_set = self.test_set = torchvision.datasets.ImageFolder(f"{self.cifar_path}/test")
            
    @staticmethod
    def collate_class_labels(x):
        return [img for img, index in x], [CIFAR10DataModule.classes[index] for img, index in x]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)


class WebdatasetDataModule(TransformableImageDataModule):
    @staticmethod
    def standard_preprocess(sample):
        image, json = sample
        label = json["caption"]
        return image, label

    def __init__(self, train_paths, val_paths, test_paths=None, collate_type=CollateType.COLLATE_NONE_TUPLE, collate_fn=None, batch_size=16, num_workers=4, img_in_target_size=64):
        super(WebdatasetDataModule, self).__init__(collate_type, collate_fn, img_in_target_size)
        self.batch_size = batch_size
        self.train_paths = self.braceexpand_paths(train_paths)
        self.val_paths = self.braceexpand_paths(val_paths)
        
        self.test_paths = test_paths
        if self.test_paths is not None:
            self.test_paths = self.braceexpand_paths(test_paths)
        
        self.num_workers = num_workers
        self.img_in_target_size = img_in_target_size
        self.transform = ImageTransformer(img_target_size=self.img_in_target_size)

    def braceexpand_paths(self, paths):
        if len(paths) == 1:
            return paths[0]

        return functools.reduce(operator.add, map(lambda a: list(braceexpand.braceexpand(a)), paths))

    def train_dataloader(self):
        ds = (wds.WebDataset(self.train_paths, shardshuffle=True)
            .shuffle(1000)
            .decode("pil")
            .to_tuple("jpg", "json")
            .map(WebdatasetDataModule.standard_preprocess)
            .batched(self.batch_size))
        return (wds.WebLoader(ds, num_workers=self.num_workers, batch_size=None, collate_fn=self.collate)
            .unbatched()
            .shuffle(1000)
            .batched(self.batch_size))

    def val_dataloader(self):
        ds = (wds.WebDataset(self.val_paths, shardshuffle=True)
            .shuffle(1000, initial=10000)
            .decode("pil").to_tuple("jpg", "json")
            .map(WebdatasetDataModule.standard_preprocess)
            .batched(self.batch_size))
        return (wds.WebLoader(ds, num_workers=self.num_workers, batch_size=None, collate_fn=self.collate)
            .unbatched()
            .shuffle(1000)
            .batched(self.batch_size))
    
    def test_dataloader(self):
        if self.test_paths is None:
            print("Warning: No test paths defined, returning a Validation Dataloader!")
            return self.val_dataloader()
        else:
            ds = (wds.WebDataset(self.test_paths, shardshuffle=True)
                .shuffle(1000, initial=10000)
                .decode("pil").to_tuple("jpg", "json")
                .map(WebdatasetDataModule.standard_preprocess)
                .batched(self.batch_size))
            return (wds.WebLoader(ds, num_workers=self.num_workers, batch_size=None, collate_fn=self.collate)
                .unbatched()
                .shuffle(1000)
                .batched(self.batch_size))

    
class VideoDatasetDataModule(TransformableDataModule):
    def __init__(
        self, 
        train_csv_path,
        train_data_path, 
        val_csv_path, 
        val_data_path, 
        test_csv_path=None, 
        test_data_path=None, 
        batch_size=4, 
        num_workers=4, 
        target_resolution=(64, 64),
        padding_value=-1,
        nth_frames=5,
        max_frames_per_part=16,
        min_frames_per_part=4,
        first_part_only=True
    ):
        super(VideoDatasetDataModule, self).__init__()
        self.batch_size = batch_size
        self.train_csv_path = train_csv_path
        self.train_data_path = train_data_path
        self.val_csv_path = val_csv_path
        self.val_data_path = val_data_path
        self.test_csv_path = test_csv_path
        self.test_data_path = test_data_path
        self.num_workers = num_workers
        self.target_resolution = target_resolution
        self.padding_value = padding_value
        self.nth_frames = nth_frames
        self.max_frames_per_part = max_frames_per_part
        self.t_data = VideoDataset(
            self.train_csv_path, 
            self.train_data_path, 
            target_resolution=self.target_resolution, 
            max_frames_per_part=self.max_frames_per_part,
            nth_frames=self.nth_frames,
            first_part_only=first_part_only,
            min_frames_per_part=min_frames_per_part,
            target_ordering="c t h w",
            normalize=False
        )
        self.v_data = VideoDataset(
            self.val_csv_path, 
            self.val_data_path, 
            target_resolution=self.target_resolution, 
            target_ordering="c t h w",
            max_frames_per_part=self.max_frames_per_part,
            nth_frames=self.nth_frames,
            first_part_only=first_part_only,
            min_frames_per_part=min_frames_per_part,
            normalize=False
        )
        
        self.test_data = None
        if self.test_csv_path is not None and self.test_data_path is not None:
            self.test_data = VideoDataset(
                self.test_csv_path, 
                self.test_data_path, 
                target_resolution=self.target_resolution, 
                target_ordering="c t h w",
                max_frames_per_part=self.max_frames_per_part,
                nth_frames=self.nth_frames,
                first_part_only=first_part_only,
                min_frames_per_part=min_frames_per_part,
                normalize=False
            )

        self.collate = lambda data: CollateTypeFunction.cntps(data, padding_value=self.padding_value)

    def train_dataloader(self):
        return DataLoader(
            self.t_data,
            batch_size=self.batch_size ,
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=False,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.v_data,
            batch_size=self.batch_size ,
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        if self.test_data is None:
            print("Warning: No test paths defined, returning a Validation Dataloader!")
            return self.val_dataloader()
        else:
            return DataLoader(
                self.test_data,
                batch_size=self.batch_size ,
                shuffle=True,
                collate_fn=self.collate,
                pin_memory=False,
                num_workers=self.num_workers
            )

    def transform_batch(self, batch):
        return self.t_data.normalize(batch)

    def reverse_transform_batch(self, batch):
        return self.t_data.reverse_normalize(batch)

    def transform(self, data):
        return self.t_data.normalize(data)

    def reverse_transform(self, data):
        return self.t_data.reverse_normalize(data)