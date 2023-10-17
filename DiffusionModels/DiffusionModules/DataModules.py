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
from torch.utils.data import DataLoader
from collections import OrderedDict


class CollateTypeFunction():
    # The standard collations translate the tuples of data into a dict where the first element of the tuple is mapped 
    # to "data" and the second element is mapped to "caption"
    standard_collations = OrderedDict(((0, "data"), (1, "caption")))
    # The standard collations with fps translate the tuples of data into a dict where the first element of the tuple is mapped
    # to "data", the second element is mapped to "caption", and the third element is mapped to "fps"
    standard_collations_with_fps = OrderedDict(((0, "data"), (1, "caption"), (2, "fps")))

    @staticmethod
    def additional_collate_none(data, collations=None):
        """
        Filters out any data where the tuple contains a None value. After that it translates the tuples of data into a dict
        where the elements of the tuple are mapped to the keys of the collations dict.

        :param data: Tuple of data each containing Images and Captions.
        :param collations: OrderedDictionary for describing the collations with a key as a tuple index and string as target dict key, 
                           defaults to OrderedDict(((0, "data"), (1, "caption")))
        :return: Dict of data defined by the collations.
        """        
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
        """
        Filters out any data where the tuple contains a None value. Translates the tuples of combined data into
        a dictionary, depending on the collations. The collations are defined by an OrderedDictionary where the key
        is the tuple index and the value is the target dict key.

        :param data: Tuples of data each containing Images and Captions.
        :param collations: OrderedDictionary for describing the collations with a key as a tuple index and string as target dict key, 
                           defaults to OrderedDict(((0, "data"), (1, "caption")))
        :return: Dict of data defined by the collations.
        """        
        data = CollateTypeFunction.additional_collate_none(data, collations)
        return data

    @staticmethod
    def cnt(data, collations=None):
        """
        Filters out any data where the tuple contains a None value. Translates the tuples of combined data into
        a tuple of multiple tensors, depending on the collations. The collations are defined by an OrderedDictionary
        where the key is the tuple index and the value is the target key of an OrderedDict that is used before
        translating the data into a tuple of data. The given collations define the order of data in the
        tuple.

        :param data: Tuples of data each containing Images and Captions.
        :param collations: OrderedDictionary for describing the collations with a key as a tuple index and string as target dict key,
                           defaults to OrderedDict(((0, "data"), (1, "caption")))
        :return: Tuple of data defined by the collations.
        """        
        data = CollateTypeFunction.additional_collate_none(data, collations)
        return tuple(data.values())
        
    @staticmethod
    def ci(data):
        """
        Identity function that returns the tuples of data without any changes.

        :param data: Tuple of data.
        :return: Tuples of data.
        """        
        return data

    @staticmethod
    def cntps(data, padding_value=-1, collations=None):
        """
        Filters out any data where the tuple contains a None value. Translates the tuples of combined data into
        a tuple of multiple tensors, depending on the collations. The collations are defined by an OrderedDictionary
        where the key is the tuple index and the value is the target key of an OrderedDict that is used before
        translating the data into a tuple of data. The given collations define the order of the tensors in the
        tuple. The data is padded to the same length with the given padding_value.

        :param data: Tuples of data each containing Videos, Captions, and FPS.
        :param padding_value: Value to pad the data with, defaults to -1
        :param collations: OrderedDictionary for describing the collations with a key as a tuple index and string as target dict key,
                           defaults to OrderedDict(((0, "data"), (1, "caption"), (2, "fps")))
        :return: Tuple of data defined by the collations.
        """        
        if collations is None:
            collations = CollateTypeFunction.standard_collations_with_fps
        data, captions, fps = CollateTypeFunction.cnt(list(zip(*data)), collations)
        lengths = torch.tensor([len(e) for e in data])
        if len(data) > 0:
            video = pad_sequence(data, padding_value=padding_value, batch_first=True)
        return video, captions, lengths, fps


collate_type_to_function = {
    "COLLATE_NONE_TUPLE": CollateTypeFunction.cnt,
    "COLLATE_NONE_DICT": CollateTypeFunction.cnd,
    "COLLATE_IDENTITY": CollateTypeFunction.ci
}

class CollateType(Enum):
    # Uses the cnt function to collate the data.
    COLLATE_NONE_TUPLE="COLLATE_NONE_TUPLE"
    # Uses the cnd function to collate the data.
    COLLATE_NONE_DICT="COLLATE_NONE_DICT"
    # Uses the ci function to collate the data.
    COLLATE_IDENTITY="COLLATE_IDENTITY"

    def __call__(self, *args):
        """
        Calls the function that is mapped to the enum value with the given arguments.

        :return: Result of the function call.
        """        
        return collate_type_to_function[self.value](*args)

class TransformableDataModule(LightningDataModule, ABC):
    @abstractmethod
    def train_dataloader(self):
        """
        Returns a dataloader for the training data.
        """        
        pass
    
    @abstractmethod
    def val_dataloader(self):
        """
        Returns a dataloader for the validation data.
        """        
        pass
    
    @abstractmethod
    def test_dataloader(self):
        """
        Returns a dataloader for the test data.
        """        
        pass

    @abstractmethod
    def transform_batch(self, batch):
        """
        Transforms a batch of data.

        :param batch: Batch of data.
        """        
        pass

    @abstractmethod
    def reverse_transform_batch(self, batch):
        """
        Reverses the transformation of a batch of data.

        :param batch: Transformed batch of data.
        """        
        pass

    @abstractmethod
    def transform(self, data):
        """
        Transforms a single element of data.

        :param data: Element of data.
        """        
        pass

    @abstractmethod
    def reverse_transform(self, data):
        """
        Reverses the transformation of a single element of data.

        :param data: Transformed element of data.
        """        
        pass


class TransformableImageDataModule(TransformableDataModule, ABC):
    def __init__(self, collate_type=CollateType.COLLATE_NONE_TUPLE, collate_fn=None, img_in_target_size=64):
        """
        Creates an ImageDataModule that can transform images. The images are transformed to the given target size
        using the ImageTransformer class, which translate numpy or PIL images to tensors and brings them to the -1 to 1 range.
        Is also able to reverse the transformation. The collate function is used to collate the data.

        :param collate_type: CollateType of the collate function, defaults to CollateType.COLLATE_NONE_TUPLE
        :param collate_fn: Custom collate function. Is set to None for predefined collates, defaults to None
        :param img_in_target_size: Target size of the images, defaults to 64
        """        
        super(TransformableImageDataModule, self).__init__()
        self.img_in_target_size = img_in_target_size
        self.transform = ImageTransformer(img_target_size=self.img_in_target_size)
        self.collate = collate_type if isinstance(collate_type, CollateType) else collate_fn

    @abstractmethod
    def train_dataloader(self):
        """
        Returns a dataloader for the training data.
        """        
        pass

    @abstractmethod
    def val_dataloader(self):
        """
        Returns a dataloader for the validation data.
        """        
        pass
    
    @abstractmethod
    def test_dataloader(self):
        """
        Returns a dataloader for the test data.
        """        
        pass

    def transform_batch(self, batch):
        """
        Transforms a batch of data using the ImageTransformer.

        :param batch: Batch of data.
        :return: Transformed batch of data.
        """        
        return self.transform.transform_images(batch)

    def reverse_transform_batch(self, batch):
        """
        Reverses the transformation of a batch of data using the ImageTransformer.

        :param batch: Transformed batch of data.
        :return: Batch of data.
        """        
        return self.transform.reverse_transform_images(batch)

    def transform(self, data):
        """
        Transforms a single element of data using the ImageTransformer.

        :param data: Element of data.
        :return: Transformed element of data.
        """        
        return self.transform.transform_image(data)

    def reverse_transform(self, data):
        """
        Reverses the transformation of a single element of data using the ImageTransformer.

        :param data: Transformed element of data.
        :return: Element of data.
        """        
        return self.transform.reverse_transform_image(data)


class CIFAR10DataModule(TransformableImageDataModule):
    classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    def __init__(self, cifar_path="/home/archive/cifar10-64", batch_size=16, separate_val_split=False, separate_val_split_ratio=0.1, num_workers=4, img_in_target_size=64):
        """
        Creates a CIFAR10DataModule that loads the CIFAR10 dataset from the given path. The dataset is split into
        training, validation, and test set. The training and validation set are loaded from the train folder and the
        test set is loaded from the test folder if separate_val_split is True. If separate_val_split is False the
        training set is loaded from the train folder and the validation set is loaded from the test folder. In that case
        no test set is provided. The target size of the images is set to the given img_in_target_size. 

        :param cifar_path: Path to the CIFAR10 dataset, defaults to "/home/archive/cifar10-64"
        :param batch_size: batch_size of the dataloaders, defaults to 16
        :param separate_val_split: Uses a validation split if True, else it uses the test set as validation set, defaults to False
        :param separate_val_split_ratio: Ratio of the validation split if separate_val_split is True, defaults to 0.1
        :param num_workers: Number of workers for the dataloaders, defaults to 4
        :param img_in_target_size: Target size of the images, defaults to 64
        """        
        super(CIFAR10DataModule, self).__init__(CollateType.COLLATE_NONE_TUPLE, None, img_in_target_size)
        self.batch_size = batch_size
        self.cifar_path = cifar_path
        self.num_workers = num_workers
        self.separate_val_split = separate_val_split
        self.separate_val_split_ratio = separate_val_split_ratio
        self.collate_base = self.collate
        self.collate = lambda x: CIFAR10DataModule.collate_class_labels(self.collate_base(list(zip(*x))))
        
        if self.separate_val_split:
            trainset = torchvision.datasets.ImageFolder(f"{self.cifar_path}/train")
            self.val_set, self.train_set = torch.utils.data.random_split(trainset, [self.separate_val_split_ratio, 1.0 - self.separate_val_split_ratio])
            self.test_set = torchvision.datasets.ImageFolder(f"{self.cifar_path}/test")
        else:
            self.train_set = torchvision.datasets.ImageFolder(f"{self.cifar_path}/train")
            self.val_set = self.test_set = torchvision.datasets.ImageFolder(f"{self.cifar_path}/test")
            
    @staticmethod
    def collate_class_labels(x):
        """
        Collates the class labels of the data.

        :param x: Data to collate.
        :return: Collated data.
        """        
        return x[0], [CIFAR10DataModule.classes[index] for index in x[1]]

    def train_dataloader(self):
        """
        Returns a dataloader for the training data.

        :return: Dataloader for the training data.
        """        
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)

    def val_dataloader(self):
        """
        Returns a dataloader for the validation data.

        :return: Dataloader for the validation data.
        """        
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)
    
    def test_dataloader(self):
        """
        Returns a dataloader for the test data. If separate_val_split is False it returns a dataloader for the validation data.

        :return: Dataloader for the test data.
        """        
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate)


class WebdatasetDataModule(TransformableImageDataModule):
    @staticmethod
    def standard_preprocess(sample):
        """
        Extract the caption from the json and return the image and caption.

        :param sample: Element of the dataset.
        :return: Image and caption.
        """        
        image, json = sample
        label = json["caption"]
        return image, label

    def __init__(self, train_paths, val_paths, test_paths=None, collate_type=CollateType.COLLATE_NONE_TUPLE, collate_fn=None, batch_size=16, num_workers=4, img_in_target_size=64):
        """
        Creates a WebdatasetDataModule that loads the dataset from the given paths. The dataset is defined by a list of paths
        for the training, validation, and test set. The dataset is loaded using the Webdataset library. The target size of the
        images is set to the given img_in_target_size. The collate function is used to collate the data. 

        :param train_paths: List of Paths to the training dataset. 
        :param val_paths: List of Paths to the validation dataset.
        :param test_paths: List of Paths to the test dataset, if None the validation dataset is used as test dataset, defaults to None
        :param collate_type: CollateType of the collate function, defaults to CollateType.COLLATE_NONE_TUPLE
        :param collate_fn: Custom collate function. Is set to None for predefined collates, defaults to None
        :param batch_size: batch_size of the dataloaders, defaults to 16
        :param num_workers: Number of workers for the dataloaders, defaults to 4
        :param img_in_target_size: Target size of the images, defaults to 64
        """        
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
        """
        Merges the paths into a single list. 

        :param paths: List of paths.
        :return: Merged list of paths.
        """        
        if len(paths) == 1:
            return paths[0]

        return functools.reduce(operator.add, map(lambda a: list(braceexpand.braceexpand(a)), paths))

    def train_dataloader(self):
        """
        Returns a dataloader for the training data.

        :return: Dataloader for the training data.
        """        
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
        """
        Returns a dataloader for the validation data.

        :return: Dataloader for the validation data.
        """        
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
        """
        Returns a dataloader for the test data. If test_paths is None it returns a dataloader for the validation data.

        :return: Dataloader for the test data.
        """        
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
        """
        Creates a VideoDatasetDataModule that loads the dataset from the given paths using the VideoDataset class. The sets are defined by a path to a csv file
        and a path to the data. If test_csv_path and test_data_path are None the test set is not defined and the validation set is used as test set. The target
        resolution of the videos is set to the given target_resolution. The padding_value is used to pad the videos to the same length. The nth_frames parameter
        is used to only use every nth frame of the videos. The max_frames_per_part parameter is used to limit the number of frames per video part. A video with
        less than min_frames_per_part after the nth_frames are skipped. The first_part_only parameter is used to only use the first part of the video if True.

        :param train_csv_path: Path to the training csv file.
        :param train_data_path: Path to the training data.
        :param val_csv_path: Path to the validation csv file.
        :param val_data_path: Path to the validation data.
        :param test_csv_path: Path to the test csv file, defaults to None
        :param test_data_path: Path to the test data, defaults to None
        :param batch_size: batch_size of the dataloaders, defaults to 4
        :param num_workers: Number of workers for the dataloaders, defaults to 4
        :param target_resolution: Target resolution of the videos, defaults to (64, 64)
        :param padding_value: Value to pad the videos with, defaults to -1
        :param nth_frames: Only uses every nth frame of the videos, defaults to 5
        :param max_frames_per_part: Limits the number of frames per video part, defaults to 16
        :param min_frames_per_part: Skips videos with less than min_frames_per_part after the nth_frames are applied, defaults to 4
        :param first_part_only: Only uses the first part of the video if True, defaults to True
        """        
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
        """
        Returns a dataloader for the training data.

        :return: Dataloader for the training data.
        """        
        return DataLoader(
            self.t_data,
            batch_size=self.batch_size ,
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=False,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        """
        Returns a dataloader for the validation data.

        :return: Dataloader for the validation data.
        """        
        return DataLoader(
            self.v_data,
            batch_size=self.batch_size ,
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=False,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        """
        Returns a dataloader for the test data. If test_csv_path and test_data_path are None it returns a dataloader for the validation data.

        :return: Dataloader for the test data.
        """        
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
        """
        Transforms a batch of data using the VideoDataset.

        :param batch: Batch of data.
        :return: Transformed batch of data.
        """        
        return self.t_data.normalize(batch)

    def reverse_transform_batch(self, batch):
        """
        Reverses the transformation of a batch of data using the VideoDataset.

        :param batch: Transformed batch of data.
        :return: Batch of data.
        """        
        return self.t_data.reverse_normalize(batch)

    def transform(self, data):
        """
        Transforms a single element of data using the VideoDataset.

        :param data: Element of data.
        :return: Transformed element of data.
        """        
        return self.t_data.normalize(data)

    def reverse_transform(self, data):
        """
        Reverses the transformation of a single element of data using the VideoDataset.

        :param data: Transformed element of data.
        :return: Element of data.
        """        
        return self.t_data.reverse_normalize(data)
