import os
from abc import ABC, abstractmethod
from einops import rearrange
import random
import torch
import clip
from torch import nn

from DiffusionModules.ClipTranslatorModules import ClipTranslatorTrainer
from DiffusionModules.DataModules import CIFAR10DataModule

class ClipTools(nn.Module): 
    def __init__(self, clip_model="ViT-B/32", device=None): 
        """
        Clip tools for working with the models implemented by OpenAI.

        :param clip_model: CLIP model to use, defaults to "ViT-B/32"
        :param device: Device to use, defaults to ("cuda" if torch.cuda.is_available() else "cpu")
        """        
        super().__init__()
        self._device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self._clip_model, self._clip_preprocess = clip.load(clip_model, device=self._device)
        self._clip_model.eval()
    
    def get_clip_emb_size(self):
        """
        Gets the CLIP embedding size.

        :return: CLIP embedding size.
        """        
        return self._clip_model.visual.output_dim 
    
    def get_clip_emb_images(self, images):
        """
        Gets the CLIP embedding for a list of PIL-images.

        :param images: List of PIL-images to get the CLIP embedding for.
        :return: CLIP embeddings for the images.
        """        
        images = torch.stack([self._clip_preprocess(i) for i in images])
        return self._clip_model.encode_image(images.to(self._device)).float()

    def get_clip_emb_videos(self, videos):
        """
        Gets the CLIP embedding for a list of videos. This ignores the temporal dimension and 
        averages the embeddings. This was only a test and is not used in the current version.

        :param videos: List of videos to get the CLIP embeddings for.
        :return: CLIP embedding for each videos.
        """        
        b, t = videos.shape[0], videos.shape[1]
        stacked_frames = rearrange(videos, 'b t c h w -> (b t) c h w')
        videos = torch.stack([self._clip_preprocess(i) for i in stacked_frames])
        embs = self._clip_model.encode_image(videos.to(self._device)).float()
        embs = rearrange(embs, '(b t) e -> b t e', b=b, t=t)
        return embs.mean(dim=1)
    
    def get_clip_emb_text(self, texts):
        """
        Gets the CLIP embedding for a list of trunctated texts.

        :param texts: Texts to get the CLIP embedding for.
        :return: CLIP embeddings for the texts.
        """        
        return self._clip_model.encode_text(clip.tokenize(texts, truncate = True).to(self._device)).float()
        
    def forward(self, images, texts):
        """
        Gets the CLIP embeddings for a list of images and texts.

        :param images: List of PIL-images to get the CLIP embeddings for.
        :param texts: Texts to get the CLIP embeddings for.
        :return: Tuple of CLIP embeddings for the images and texts.
        """        
        return self.get_clip_emb_images(images), self.get_clip_emb_text(texts)

class BaseEmbeddingProvider(ABC, nn.Module):
    @abstractmethod
    def get_embedding(self, images, labels):
        """
        Returns the embedding for a list of items and labels.

        :param items: List of items to get the embedding for.
        :param labels: List of labels to get the embedding for.
        """        
        pass

    def forward(self, items, labels):
        """
        Returns the embedding for a list of items and labels.

        :param items: List of items to get the embedding for.
        :param labels: List of labels to get the embedding for.
        :return: Embedding for the items and labels.
        """        
        return self.get_embedding(images, labels)

class ClipRandomImageTextTranslatorEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, translator_model_path, clip_tools=None):
        """
        Embedding provider that randomly selects between the ClipTranslatorEmbeddingProvider
        ClipTextEmbeddingProvider and ClipEmbeddingProvider. 

        :param translator_model_path: Path to the translator model.
        :param clip_tools: ClipTools instance to use, defaults to a new basic instance.
        """        
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools
        self.clip_translator_provider = ClipTranslatorEmbeddingProvider(translator_model_path, self.clip_tools)
        self.clip_image_provider = ClipEmbeddingProvider(self.clip_tools)
        self.clip_text_provider = ClipTextEmbeddingProvider(self.clip_tools)
        self.providers = [self.clip_translator_provider, self.clip_image_provider, self.clip_text_provider]

    def get_embedding(self, items, labels):
        return random.choice(self.providers).get_embedding(items, labels)

class ClipTranslatorEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, translator_model_path, clip_tools=None):
        """
        Embedding provider that uses the ClipTranslatorEmbeddingProvider to get translator embeddings.

        :param translator_model_path: Path to the translator model.
        :param clip_tools: ClipTools instance to use, defaults to a new basic instance.
        """        
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools
        self.translator_model_path = translator_model_path
        self.model = ClipTranslatorTrainer.load_from_checkpoint(self.translator_model_path, map_location=self.clip_tools._device).model
        self.model.eval()
        self.model.to(self.clip_tools._device)

    def get_embedding(self, items, labels):
        """
        Gets the translator embeddings for a list labels. This
        ignores the images and only uses the labels.

        :param items: The images are ignored and not used. This can be None.
        :param labels: List of labels to get the translator embeddings for.
        :return: Translator embeddings for the labels.
        """        
        cap_emb = self.clip_tools.get_clip_emb_text(labels)
        i_embs = self.model(cap_emb)
        return i_embs
    
class ClipEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, clip_tools=None):
        """
        Clip embedding provider that uses the ClipTools to get image embeddings.

        :param clip_tools: ClipTools instance to use, defaults to a new basic instance.
        """        
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools
        
    def get_embedding(self, items, labels):
        """
        Gets the CLIP embeddings for a list of images. This ignores the labels and
        only uses the images.

        :param items: List of images to get the CLIP embeddings for.
        :param labels: The labels are ignored and not used. This can be None.
        :return: CLIP embeddings for the images.
        """
        i_embs = self.clip_tools.get_clip_emb_images(items)    
        return i_embs
    

class ClipTextEmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, clip_tools=None):
        """
        Clip embedding provider that uses the ClipTools to get text embeddings.

        :param clip_tools: ClipTools instance to use, defaults to a new basic instance.
        """        
        super().__init__()
        self.clip_tools = ClipTools() if clip_tools is None else clip_tools

    def get_embedding(self, items, labels):
        """
        Gets the CLIP embeddings for a list of texts. This ignores the images and
        only uses the texts.

        :param items: The images are ignored and not used. This can be None.
        :param labels: List of texts to get the CLIP embeddings for.
        :return: CLIP embeddings for the texts.
        """        
        v_embs = self.clip_tools.get_clip_emb_text(labels)    
        return v_embs

    
class CF10EmbeddingProvider(BaseEmbeddingProvider):
    def __init__(self, classes=None):
        """
        Embedding provider that uses the given classes to get one-hot encodings.

        :param classes: List of classes to use, defaults to CIFAR10DataModule.classes.
        """        
        super().__init__()
        self.classes = classes if classes is not None else CIFAR10DataModule.classes
        self.num_classes = len(self.classes)
        
    def get_embedding(self, images, labels):
        """
        Gets the one-hot encodings for a list of labels.

        :param images: The images are ignored and not used. This can be None.
        :param labels: List of labels to get the one-hot encodings for. Must be in self.classes.
        :return: One-hot encodings for the labels.
        """        
        # pylint: disable=E1102
        labels = [self.classes.index(label) for label in labels]
        labels = torch.Tensor(labels).long()
        return nn.functional.one_hot(labels, self.num_classes).float()