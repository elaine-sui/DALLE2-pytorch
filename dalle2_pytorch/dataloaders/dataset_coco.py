import torch
import os
import pickle
import sys
import json
import random
from typing import Tuple, Optional, Union

import skimage.io as io
import clip
from PIL import Image

from omegaconf import OmegaConf
from torch.utils.data import Dataset

from ..enums import Modality
TEXT_TO_IMG_GAP_PATH = '/pasteur/u/esui/data/coco/normalized_cap_to_img_gap.pkl'
TEXT_EMBED_MEAN = '/pasteur/u/esui/data/coco/normalized_text_embed_mean.pkl'
IMAGE_EMBED_MEAN = '/pasteur/u/esui/data/coco/normalized_image_embed_mean.pkl'

# Mainly copy-paste from captioning dataset (note: tokenizer not needed)
class CocoDataset(Dataset):

    def __init__(self, cfg, split='train', transform=None):
        
        print("="*80)
        print("Data split: ", split)
        print("="*80)
        
        self.split = split
        self.transform = transform
        
        self.cfg = cfg
        self.remove_modality_gap = self.cfg.data.remove_modality_gap
        self.remove_mean = self.cfg.data.remove_mean
        
        data_path = self.get_data_path(cfg, split)
        self.normalize_embed = cfg.data.normalize_embed
        
        ###################
        print("=> Loading all_data pkl")
        with open(data_path, 'rb') as f:
            all_data = pickle.load(f)
        print("Number of images is %0d" % len(all_data["images"]))
        print("Number of captions is %0d" % len(all_data["captions"]))
        sys.stdout.flush()
        
        # {image_id: {"img_path": ..., "embed": ...}}
        self.images = all_data["images"]
        # {caption_id: {"caption": .., "img_id": .., "embed": ...}}
        self.captions = all_data["captions"]
        
        ###################
        
        if os.path.isfile(f"{data_path[:-4]}_tokens.pkl"):
            print("=> Loading caption_id_2_image_id, captions_tokens, all_len dicts")
            with open(f"{data_path[:-4]}_tokens.pkl", 'rb') as f:
                self.captions_tokens, self.caption_id_2_image_id, all_len = pickle.load(f)
        else:
            # {caption_id: image_id}
            print("=> Saving caption_id_2_image_id dict")
            self.caption_id_2_image_id = {sentid: self.captions[sentid]["img_id"] for sentid in self.captions}
            # {caption_id: tokenizer(caption)}
            print("=> Saving captions_tokens dict")
            self.captions_tokens = None
            print("=> Saving all_len dict")
            all_len = None
            
            with open(f"{data_path[:-4]}_tokens.pkl", 'wb') as f:
                pickle.dump([self.captions_tokens, self.caption_id_2_image_id, all_len], f)
    
        self.output_modality = Modality.Vision
        
        # In testing, input modality must be opposite of output modality to evaluate cross-modal task
        if self.split == "test":
            self.input_modality = Modality.Language
        else:
            self.input_modality = self.cfg.encoder.modality
        
        # Get all caption and image ids
        self.img_ids = sorted(list(self.images.keys()))
        random.shuffle(self.img_ids)
        self.cap_ids = sorted(list(self.captions.keys()))
        random.shuffle(self.cap_ids)
        
        # Sample data
        if "train" in self.split and not OmegaConf.is_none(cfg.data, 'sample_frac'):
            img_sample_size = int(len(self.img_ids) * cfg.data.sample_frac)
            cap_sample_size = int(len(self.cap_ids) * cfg.data.sample_frac)
            self.img_ids = random.sample(self.img_ids, img_sample_size)
            self.cap_ids = random.sample(self.cap_ids, cap_sample_size)
        
        # Load modality gap
        with open(TEXT_TO_IMG_GAP_PATH, 'rb') as f:
            self.text_to_img_modality_gap = pickle.load(f)
            
        # Load means gap
        if self.input_modality == Modality.Language:
            with open(TEXT_EMBED_MEAN, 'rb') as f:
                self.embed_mean = pickle.load(f)
        else:
            with open(IMAGE_EMBED_MEAN, 'rb') as f:
                self.embed = pickle.load(f)
        
        ## Preprocess image to clip
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # _, self.preprocess_img = clip.load(self.cfg.encoder.clip_model_type, 
        #                                    device=device, jit=False)
        
        
        
    def get_data_path(self, cfg, split):
        if split == 'train':
            data_path = cfg.data.train_data_path
        elif split == 'val':
            data_path = cfg.data.val_data_path
        elif split == 'test':
            data_path = cfg.data.test_data_path
        elif split == 'restval':
            data_path = cfg.data.restval_data_path
        elif split == 'train+restval':
            data_path = cfg.data.train_restval_data_path
        else:
            raise NotImplementedError(f"split {split} invalid")
            
        return data_path
    
    def __len__(self) -> int:
        if (self.input_modality == Modality.Vision and \
              self.output_modality == Modality.Vision):
            # Image reconstruction
            return len(self.img_ids)
        else:
            return len(self.cap_ids)
    
    def __getitem__(self, item: int) -> Tuple[torch.Tensor, ...]:
        # (img, emb, txt), where emb = {"img": ...}
        
        item = self.cap_ids[item]
        img_id = self.caption_id_2_image_id[item]
        
        if self.input_modality == Modality.Vision:
            embed = self.images[img_id]["embed"]
        else:
            embed = self.images[img_id]["embed"]
        
        if self.normalize_embed:
            embed = embed.float()
            embed = embed / embed.norm(2, -1)
        
        if self.remove_modality_gap and self.input_modality == Modality.Language:
            # note: the gap was computed as img - text
            embed += self.text_to_img_modality_gap 
        elif self.remove_mean:
            embed -= self.embed_mean
            
        # always set the embedding as "image" for the sake of training the decoder
        embed = {"img": embed.squeeze()} 
        
        img_path = self.images[img_id]["img_path"]
        image = io.imread(img_path)
        image = self.transform(Image.fromarray(image))
        
        if image.shape[0] != 3:
            image = image.repeat((3, 1, 1))
        
        # image = self.preprocess_img(Image.fromarray(image))
        
        txt = self.captions[item]['caption']
        
        return (image, embed, txt)
    
## To get stuff:
# image_path = self.images[img_id]["img_path"]
# image_embed = self.images[img_id]["embed"]
# caption = self.captions[sent_id]["caption"]
# image_id_for_caption = self.captions[sent_id]["img_id"]
# caption_embed = self.captions[sent_id]["embed"]
