import random
import pickle
import os
from torch.utils.data import Dataset
from dalle2_pytorch.tokenizer import tokenizer

from .constants import DATA_PATH, NUM_SAMPLES, NUM_REPEATS, AVG_OUTPUT_EMBED_FOLDER

# Basic dataset for captions
class CocoDataset(Dataset):

    def __init__(self, num_samples=500, num_repeats=100):
        with open(DATA_PATH, 'rb') as f:
            all_data = pickle.load(f)

        self.captions = all_data["captions"]

        self.cap_ids = sorted(list(self.captions.keys()))
        random.shuffle(self.cap_ids)
        
        self.cap_ids = self.cap_ids[:num_samples]
        
        self.num_repeats = num_repeats
    
    def __len__(self):
        return len(self.cap_ids)

    def __getitem__(self, item):
        cap_id = self.cap_ids[item]
        caption = self.captions[cap_id]['caption']
        
        prior_text_input = [caption] * self.num_repeats
        tokens = tokenizer.tokenize(prior_text_input)
        
        return cap_id, tokens
    

def build_dataset():
    return CocoDataset(NUM_SAMPLES, NUM_REPEATS)