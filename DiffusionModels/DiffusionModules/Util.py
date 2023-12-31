import torch
import numpy as np
import torchvision.transforms as transforms
import io
import re
import requests
import html
import hashlib
import urllib
import urllib.request

from typing import Any

def open_url(url: str, num_attempts: int = 10, verbose: bool = True, return_filename: bool = False) -> Any:
    """
    This was taken from: https://github.com/universome/fvd-comparison/blob/master/util.py

    Download the given URL and return a binary-mode file object to access the data.
    """
    assert num_attempts >= 1

    # Doesn't look like an URL scheme so interpret it as a local filename.
    if not re.match('^[a-z]+://', url):
        return url if return_filename else open(url, "rb")

    # Handle file URLs.  This code handles unusual file:// patterns that
    # arise on Windows:
    #
    # file:///c:/foo.txt
    #
    # which would translate to a local '/c:/foo.txt' filename that's
    # invalid.  Drop the forward slash for such pathnames.
    #
    # If you touch this code path, you should test it on both Linux and
    # Windows.
    #
    # Some internet resources suggest using urllib.request.url2pathname() but
    # but that converts forward slashes to backslashes and this causes
    # its own set of problems.
    if url.startswith('file://'):
        filename = urllib.parse.urlparse(url).path
        if re.match(r'^/[a-zA-Z]:', filename):
            filename = filename[1:]
        return filename if return_filename else open(filename, "rb")

    url_md5 = hashlib.md5(url.encode("utf-8")).hexdigest()

    # Download.
    url_name = None
    url_data = None
    with requests.Session() as session:
        if verbose:
            print("Downloading %s ..." % url, end="", flush=True)
        for attempts_left in reversed(range(num_attempts)):
            try:
                with session.get(url) as res:
                    res.raise_for_status()
                    if len(res.content) == 0:
                        raise IOError("No data received")

                    if len(res.content) < 8192:
                        content_str = res.content.decode("utf-8")
                        if "download_warning" in res.headers.get("Set-Cookie", ""):
                            links = [html.unescape(link) for link in content_str.split('"') if "export=download" in link]
                            if len(links) == 1:
                                url = requests.compat.urljoin(url, links[0])
                                raise IOError("Google Drive virus checker nag")
                        if "Google Drive - Quota exceeded" in content_str:
                            raise IOError("Google Drive download quota exceeded -- please try again later")

                    match = re.search(r'filename="([^"]*)"', res.headers.get("Content-Disposition", ""))
                    url_name = match[1] if match else url
                    url_data = res.content
                    if verbose:
                        print(" done")
                    break
            except KeyboardInterrupt:
                raise
            except:
                if not attempts_left:
                    if verbose:
                        print(" failed")
                    raise
                if verbose:
                    print(".", end="", flush=True)

    # Return data as file object.
    assert not return_filename
    return io.BytesIO(url_data)


class ImageTransformer():
    def __init__(self, transform_image=None, reverse_transform_image=None, img_target_size=64):
        """
        Creates an ImageTransformer object for transforming images.

        :param transform_image: Pipeline for transforming images, defaults to [ToTensor, Resize(target_size), CenterCrop(target_size), Lambda(lambda t: (t * 2) - 1)]
        :param reverse_transform_image: Pipeline for reverse transforming images, 
            defaults to [Lambda(lambda t: (t + 1) / 2), Lambda(lambda t: t.permute(1, 2, 0)), Lambda(lambda t: t * 255.), Lambda(lambda t: t.numpy().astype(np.uint8)), ToPILImage()]
        :param img_target_size: Target size for images, defaults to 64
        """        
        if transform_image is None:
            # This can be used to transform a single image
            self.transform_image = transforms.Compose([
                transforms.ToTensor(), # turn into torch Tensor of shape CHW, divide by 255
                transforms.Resize(img_target_size),
                transforms.CenterCrop(img_target_size),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ])
        else:
            self.transform_image = transform_image
            
        if reverse_transform_image is None:
            # This can be used to reverse transform a single image
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
        """
        Transforms a batch of images.

        :param images: Batch of images.
        :return: Transformed batch of images.
        """        
        return torch.stack([self.transform_image(image) for image in images])
    
    def reverse_transform_images(self, tensors):
        """
        Reverse transforms a batch of images.

        :param tensors: Batch of images.
        :return: Reverse transformed batch of images.
        """        
        return [self.reverse_transform_image(tensor) for tensor in tensors]

def calculate_mean(items, key):
    """
    Calculates the mean of a list of dictionaries with the given key for each dictionary.

    :param items: List of dictionaries to calculate the mean for a key.
    :param key: Key to use for the mean calculation.
    :return: Mean of the given key for each dictionary in the list.
    """    
    sum_d = 0
    for item in items:
        sum_d += item[key]

    return sum_d / len(items)