import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torchvision import transforms as T

from . import dataloaders
from . import Decoder, Tracker, create_logger, create_saver, create_loader


def build_data_module(cfg):
    return dataloaders.data_module.DataModule(cfg)


def build_dataset(cfg):
    if cfg.data.dataset.lower() in dataloaders.ALL_DATASETS:
        return dataloaders.ALL_DATASETS[cfg.data.dataset.lower()]
    else:
        raise NotImplementedError(
            f"Dataset not implemented for {cfg.data.dataset.lower()}"
        )
        
def build_decoder(cfg):
    return Decoder(**cfg)

def build_tracker(cfg, dummy_mode):
    
    tracker = Tracker(data_path=cfg.tracker.data_path, 
                      overwrite_data_path=cfg.tracker.overwrite_data_path, 
                      dummy_mode=dummy_mode)
    
    logger = None
    if not OmegaConf.is_none(cfg.tracker, 'log'):
        logger = build_logger(cfg.tracker)
        tracker.add_logger(logger)
    if not OmegaConf.is_none(cfg.tracker, 'save'):
        saver = build_saver(cfg.tracker, logger)
        tracker.add_saver(saver)
    if not OmegaConf.is_none(cfg.tracker, 'load'):
        loader = build_loader(cfg.tracker, logger)
        tracker.add_loader(loader)
    tracker.init(cfg, extra_config={})
    return tracker


def build_logger(cfg):
    logger_type = cfg.log.pop('log_type')
    data_path = cfg.data_path
    logger = create_logger(logger_type, data_path, **cfg.log)
    
    return logger

def build_saver(cfg, logger):
    saver_type = cfg.save.pop('save_to')
    data_path = cfg.data_path
    saver = create_saver(saver_type, data_path, **cfg.save)
    return saver

def build_loader(cfg, logger):
    loader_type = cfg.load.pop('load_from')
    data_path = cfg.data_path
    loader = create_loader(loader_type, data_path, **cfg.load)
    return loader
    
    
def build_transformation(cfg):
    if OmegaConf.is_none(cfg.data, 'preprocessing'):
        return None
    elif cfg.data.preprocessing == "default":
        transforms_lst = []
        transforms_lst.append(T.RandomResizedCrop(size=[64, 64], scale=[0.75, 1.0], ratio=[1.0, 1.0]))
        transforms_lst.append(T.ToTensor())
        transforms_lst = T.Compose(transforms_lst)        
    else:
        raise NotImplementedError(f"transform {cfg.data.preprocessing} not implemented!")
    
    return transforms_lst
                              
        
    
    