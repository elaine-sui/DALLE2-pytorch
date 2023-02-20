import os
import pickle
import torch
import pytorch_lightning as pl
from .diffusion_prior import build_prior
from .constants import PRIOR_COND_SCALE, AVG_OUTPUT_EMBED_FOLDER

class DiffusionPriorLightningModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.diffusion_prior = build_prior()
    
    def configure_optimizers(self):
        return
    
    def predict_step(self, batch, batch_idx):
        rank = self.global_rank
        
        output_embed_path = os.path.join(AVG_OUTPUT_EMBED_FOLDER, f"rank{rank}_missing.pkl")
        
        if os.path.exists(output_embed_path):
            with open(output_embed_path, 'rb') as f:
                avg_embeds = pickle.load(f)["diffusion_prior_outputs"]
        else:
            avg_embeds = {}
        
        cap_ids, tokens = batch
        bs, num_repeats, token_dim = tokens.shape
        tokens_stacked = tokens.reshape(-1, token_dim) # (bs * num_repeats, token_dim)
        output_embeds = self.diffusion_prior.sample(tokens_stacked, cond_scale=PRIOR_COND_SCALE) 
        output_embeds_unstacked = output_embeds.reshape(bs, num_repeats, output_embeds.shape[-1]).cpu() 

        avg = torch.mean(output_embeds_unstacked, dim=1).cpu() # (bs, embed_dim)
        
        for i, cap_id in enumerate(cap_ids):
            cap_id = cap_id.item()
            avg_embeds[cap_id] = {"all_embeds": output_embeds_unstacked[i], "avg_embed": avg[i]}
        
        with open(output_embed_path, 'wb') as f:
            pickle.dump({"diffusion_prior_outputs": avg_embeds}, f)
        
        
        