import torch
import pickle
from torch.utils.data import DataLoader
from torchvision import transforms as T

import matplotlib.pyplot as plt
import skimage.io as io
from PIL import Image
from tqdm import tqdm
import argparse
import numpy as np

from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter
from dalle2_pytorch.tokenizer import tokenizer

from dataset_coco import CocoDataset

from constants import PRIOR_PATH, DATA_PATH, OUTPUT_EMBED_PATH, AVG_OUTPUT_EMBED_PATH, L2_DISTANCE_FROM_AVG_SAMPLED, L2_DISTANCE_BETWEEN_INPUT_OUTPUT, prior_cond_scale, device
        
def compute_input_output_distances(diffusion_prior_output):
    
    print("diffusion_prior_output:", diffusion_prior_output)
    
    if diffusion_prior_output:
        with open(OUTPUT_EMBED_PATH, 'rb') as f:
            output_data = pickle.load(f)["diffusion_prior_outputs"]
    else: # true clip image embd
        with open(OUTPUT_EMBED_PATH, 'rb') as f:
            output_data_keys = pickle.load(f)["diffusion_prior_outputs"].keys()
        
        with open(DATA_PATH, 'rb') as f:
            output_data = pickle.load(f)["images"]

        with open(f"{DATA_PATH[:-4]}_tokens.pkl", 'rb') as f:
            _, caption_id_2_image_id, _ = pickle.load(f)
        
    with open(DATA_PATH, 'rb') as f:
        input_data = pickle.load(f)["captions"]
    
    total_diff = torch.zeros(768)
    total_l2_distance = 0.
    
    l2_distances = {}
    
    total_samples = len(output_data) if diffusion_prior_output else len(output_data_keys)
    dict_to_iterate_over = output_data if diffusion_prior_output else output_data_keys
    
    for cap_id in dict_to_iterate_over:
        if diffusion_prior_output:
            output_embed = output_data[cap_id]['embed']
        else:
            img_id = caption_id_2_image_id[cap_id]
            output_embed = output_data[img_id]['embed'].numpy()
             
        input_embed = input_data[cap_id]['embed'].numpy()
        diff = output_embed - input_embed
        l2_distance = np.linalg.norm(diff)
        total_l2_distance += l2_distance
        
        l2_distances[cap_id] = l2_distance
    
    avg_l2 = total_l2_distance / total_samples
    print("avg_l2:", avg_l2)

    
def compute_sampling_distances():
    # {cap_id: {"embed": embed, "embeds": [...]}}
    
    with open(AVG_OUTPUT_EMBED_PATH, 'rb') as f:
        avg_output_data = pickle.load(f)["diffusion_prior_outputs"]
    
    l2_distances = []
    
    for cap_id in avg_output_data:
        avg_output = avg_output_data[cap_id]
        for i in len(output_data[cap_id]["embeds"]):
            sampled_output = output_data[cap_id]["embeds"][i]
        
            l2_distances.append(np.linalg.norm(avg_output - sampled_output))
            
    avg_distance = np.array(list(l2_distances.values())).mean()
    print("L2 distance between sampled and avg output:", avg_distance)
        
if __name__ == '__main__':
    diffusion_prior = build_prior()
    dataset = build_dataset()
    transform = build_transform()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute_output_embeds", action="store_true")
    parser.add_argument("--compute_avg_output_embeds", action="store_true")
    parser.add_argument("--compute_input_output_distances", action="store_true")
    parser.add_argument("--compute_sampling_distances", action="store_true")
    parser.add_argument("--not_diffusion_prior_output", action="store_true", help="when computing input output distances, if output is diffusion prior output vs. true clip image embed")
    
    args = parser.parse_args()
    
    if args.compute_output_embeds:
        compute_output_embeds(diffusion_prior, dataset)
    
    if args.compute_avg_output_embeds:
        compute_avg_output_embeds(diffusion_prior, dataset)
    
    if args.compute_input_output_distances:
        compute_input_output_distances(not args.not_diffusion_prior_output)
    
    if args.compute_sampling_distances:
        compute_sampling_distances()
    