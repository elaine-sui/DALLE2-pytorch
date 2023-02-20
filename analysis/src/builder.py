import torch
import pickle
from torchvision import transforms as T

from .dataset_coco_cap import CocoDataset

from .lightning_module import DiffusionPriorLightningModel
from .data_module import DataModule

def build_data_module(batch_size):
    return DataModule(batch_size)

def build_lightning_model():
    return DiffusionPriorLightningModel()
    
def build_transform():
    transforms_lst = []
    transforms_lst.append(T.RandomResizedCrop(size=[64, 64], scale=[0.75, 1.0], ratio=[1.0, 1.0]))
    transforms_lst.append(T.ToTensor())
    transforms_lst = T.Compose(transforms_lst)        
    
    return transforms_lst