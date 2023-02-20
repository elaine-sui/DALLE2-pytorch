import torch
import pickle
import random
from pytorch_lightning import seed_everything
from dalle2_pytorch.tokenizer import tokenizer

from constants import DATA_PATH, AVG_OUTPUT_EMBED_PATH, device

seed_everything(1234)

def compute_avg_output_embeds(diffusion_prior, dataset, num_samples=500, num_repeats=100):
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
        
    captions = all_data["captions"]
    
    cap_ids = sorted(list(captions.keys()))
    random.shuffle(cap_ids)
    
    # {cap_id: {"embed": embed}}
    diffusion_prior_out = {}
    
    for i in range(num_samples):
        cap_id = cap_ids[i]
        prior_text_input = captions[cap_id]['caption']
        # Output embeds
        tokens = tokenizer.tokenize(prior_text_input).to(device)
        output_embeds = []
        with torch.no_grad():
            for j in range(num_repeats):
                print(f"Sample {i}; Repeat {j}")
                output_embeds.append(diffusion_prior.sample(tokens, cond_scale = prior_cond_scale).cpu().squeeze())
            
            output_embeds = torch.stack(output_embeds, dim=0)
            avg = torch.mean(output_embeds, dim=0)
            
        diffusion_prior_out[cap_id] = {"embeds": output_embeds, "embed": avg}
        
        if (i + 1) % (num_samples // 10) == 0:
            with open(AVG_OUTPUT_EMBED_PATH, 'wb') as f:
                pickle.dump({"diffusion_prior_outputs": diffusion_prior_out}, f)
    
    with open(AVG_OUTPUT_EMBED_PATH, 'wb') as f:
        pickle.dump({"diffusion_prior_outputs": diffusion_prior_out}, f)

        
def compute_output_embeds(diffusion_prior, num_samples=500):
    
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
        
    captions = all_data["captions"]
    
    cap_ids = sorted(list(captions.keys()))
    random.shuffle(cap_ids)
    
    # {cap_id: {"embed": embed}}
    diffusion_prior_out = {}
    
    for i in range(num_samples):
        cap_id = cap_ids[i]
        prior_text_input = captions[cap_id]['caption']
        print(i)
        # Output embeds
        tokens = tokenizer.tokenize(prior_text_input).to(device)
        with torch.no_grad():
            output_embed = diffusion_prior.sample(tokens, cond_scale = prior_cond_scale).cpu().numpy()
            
        diffusion_prior_out[cap_id] = {"embed": output_embed.squeeze()}
        
        if (i + 1) % (num_samples // 10) == 0:
            with open(OUTPUT_EMBED_PATH, 'wb') as f:
                pickle.dump({"diffusion_prior_outputs": diffusion_prior_out}, f)
    
    with open(OUTPUT_EMBED_PATH, 'wb') as f:
        pickle.dump({"diffusion_prior_outputs": diffusion_prior_out}, f)