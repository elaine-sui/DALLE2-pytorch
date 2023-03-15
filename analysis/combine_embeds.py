from glob import glob
from pathlib import Path
import re
import pickle
import os
import torch
import torch.nn.functional as F
# from sklearn.manifold import TSNE
import argparse

from src.constants import AVG_OUTPUT_EMBED_FOLDER, COMBINED_PATH, DATA_PATH

def get_parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--combine_embeds", action="store_true")
    parser.add_argument("--reformat_features", action="store_true")
    parser.add_argument("--reformat_features_all", action="store_true")
    
    args = parser.parse_args()
    return args

def combine_embeds():
    combined_path = COMBINED_PATH
    
    combined_embeds = {}
    
    files = Path(AVG_OUTPUT_EMBED_FOLDER).glob('rank*.pkl')
    
    for filepath in files:
        print(f"Loading file: {filepath}")
        
        with open(filepath, 'rb') as f:
            rank_embeds = pickle.load(f)['diffusion_prior_outputs']
            
        for cap_id in rank_embeds:
            combined_embeds[cap_id] = {}
            combined_embeds[cap_id]['all_embeds'] = rank_embeds[cap_id]['all_embeds'].cpu()

            combined_embeds[cap_id]['avg_embed'] = rank_embeds[cap_id]['avg_embed'].cpu()
        
    with open(combined_path, 'wb') as f:
        pickle.dump({'diffusion_prior_outputs': combined_embeds}, f)
    
    print("len(combined_embeds)", len(combined_embeds))


def reformat_features_all():
    features = {}
    features_all_path = os.path.join(AVG_OUTPUT_EMBED_FOLDER, 'features_all.pt')
    
    with open(DATA_PATH, 'rb') as f:
        all_data = pickle.load(f)['captions']
    
    with open(COMBINED_PATH, 'rb') as f:
        prior_output_data = pickle.load(f)['diffusion_prior_outputs']
    
    all_image_embeds, all_text_embeds, labels = [], [], []
    
    for cap_id in prior_output_data:
        image_embed = prior_output_data[cap_id]['all_embeds'] # (50 x 768)
        text_embed = all_data[cap_id]['embed'].squeeze() # (768)
        
        all_image_embeds.append(image_embed) 
        all_text_embeds.append(text_embed)
        labels.append(cap_id)
    
    all_image_embeds = torch.stack(all_image_embeds) # (100 x 50 x 768)
    all_text_embeds = torch.vstack(all_text_embeds)
    
    all_image_embeds = all_image_embeds / torch.norm(all_image_embeds, dim=-1, keepdim=True)
    all_text_embeds = all_text_embeds / torch.norm(all_text_embeds, dim=-1, keepdim=True)
    
    labels = torch.as_tensor(labels)
    
    features["image_features"] = all_image_embeds
    features["text_features"] = all_text_embeds
    features["labels"] = labels
    
    with open(features_all_path, 'wb') as f:
        torch.save(features, f)
    
    
def reformat_features():
    features = {}
    features_path = os.path.join(AVG_OUTPUT_EMBED_FOLDER, 'features.pt')
    
    with open(DATA_PATH, 'rb') as f:
        all_data = pickle.load(f)['captions']
    
    with open(COMBINED_PATH, 'rb') as f:
        prior_output_data = pickle.load(f)['diffusion_prior_outputs']
    
    all_image_embeds, all_text_embeds, labels = [], [], []
    
    for cap_id in prior_output_data:
        image_embed = prior_output_data[cap_id]['avg_embed'].squeeze()
        text_embed = all_data[cap_id]['embed'].squeeze()
        
        all_image_embeds.append(image_embed)
        all_text_embeds.append(text_embed)
        labels.append(cap_id)
    
    all_image_embeds = torch.vstack(all_image_embeds)
    all_text_embeds = torch.vstack(all_text_embeds)
    
    all_image_embeds = all_image_embeds / torch.norm(all_image_embeds, dim=-1, keepdim=True)
    all_text_embeds = all_text_embeds / torch.norm(all_text_embeds, dim=-1, keepdim=True)
    
    labels = torch.as_tensor(labels)
    
    features["image_features"] = all_image_embeds
    features["text_features"] = all_text_embeds
    features["labels"] = labels
    
    with open(features_path, 'wb') as f:
        torch.save(features, f)
    
        
# Note this is to map the tensors to cpu (if they are currently mapped to different cuda devices)
def remap_to_cpu():
    files = Path(AVG_OUTPUT_EMBED_FOLDER).glob('rank*.pkl')
    
    for filepath in files:
        print(f"Loading file: {filepath}")
        
        with open(filepath, 'rb') as f:
            rank_embeds = pickle.load(f)['diffusion_prior_outputs']
            
        for cap_id in rank_embeds:
            rank_embeds[cap_id]['all_embeds'] = rank_embeds[cap_id]['all_embeds'].cpu()
            rank_embeds[cap_id]['avg_embed'] = rank_embeds[cap_id]['avg_embed'].cpu()
        
        with open(filepath, 'wb') as f:
            pickle.dump({'diffusion_prior_outputs': rank_embeds}, f)
    
    
        
if __name__ == '__main__':
    args = get_parser_args()
    
    if args.combine_embeds:
        combine_embeds()
    
    if args.reformat_features:
        reformat_features()
    
    if args.reformat_features_all:
        reformat_features_all()
            
        
        
            
