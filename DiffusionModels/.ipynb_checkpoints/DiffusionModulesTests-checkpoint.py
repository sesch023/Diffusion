import torch
import torch.nn as nn
from torchinfo import summary
import math
import clip
from abc import ABC, abstractmethod
from DiffusionModules import *

def test_res_block():  
    model = ResBlock(256, 256)
    batch_size = 16
    summary(model, input_size=[(batch_size, 256, 64, 64), (batch_size, 256), (batch_size, 512)])

    model = ResBlock(256, 128)
    batch_size = 16
    summary(model, input_size=[(batch_size, 256, 64, 64), (batch_size, 256), (batch_size, 512)])

    model = ResBlock(128, 256)
    batch_size = 16
    summary(model, input_size=[(batch_size, 128, 64, 64), (batch_size, 256), (batch_size, 512)])
    
    
def test_self_attention():
    model = SelfAttention(256, 64)
    batch_size = 16
    summary(model, input_size=(batch_size, 256, 64, 64))
    
    
def test_downsample():
    model = DownsampleResBlock(256, 128)
    batch_size = 16
    summary(model, input_size=[(batch_size, 256, 64, 64), (batch_size, 256), (batch_size, 512)])

    
def test_upsample():
    model = UpsampleResBlock(256, 128)
    batch_size = 16
    summary(model, input_size=[(batch_size, 256, 32, 32), (batch_size, 256), (batch_size, 512)])
    
    
def test_self_attention_res():
    model = SelfAttentionResBlock(256, 128, 32)
    batch_size = 16
    summary(model, input_size=[(batch_size, 256, 32, 32), (batch_size, 256), (batch_size, 512)])