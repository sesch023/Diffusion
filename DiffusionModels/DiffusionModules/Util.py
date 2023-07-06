import torch
import numpy as np
import torchvision.transforms as transforms

class ImageTransformer():
    def __init__(self, transform_image=None, reverse_transform_image=None, img_target_size=64):
        if transform_image is None:
            self.transform_image = transforms.Compose([
                transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
                transforms.Resize(img_target_size),
                transforms.CenterCrop(img_target_size),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transform_image = transform_image
            
        if reverse_transform_image is None:
            self.reverse_transform_image = transforms.Compose([
                transforms.Lambda(lambda t: (t + 1) / 2),
                transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
                transforms.Lambda(lambda t: t * 255.),
                transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
                transforms.ToPILImage()
            ])
        else:
            self.reverse_transform_image = reverse_transform_image
            
    def transform_images(self, images):
        return torch.stack([self.transform_image(image) for image in images])
    
    def reverse_transform_images(self, tensors):
        return [self.reverse_transform_image(tensor) for tensor in tensors]