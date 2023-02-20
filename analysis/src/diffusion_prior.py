import torch
from dalle2_pytorch import DiffusionPrior, DiffusionPriorNetwork, OpenAIClipAdapter

from .constants import PRIOR_PATH

def build_prior():

    prior_network = DiffusionPriorNetwork(
        dim=768,
        depth=24,
        dim_head=64,
        heads=32,
        normformer=True,
        attn_dropout=5e-2,
        ff_dropout=5e-2,
        num_time_embeds=1,
        num_image_embeds=1,
        num_text_embeds=1,
        num_timesteps=1000,
        ff_mult=4
    )

    diffusion_prior = DiffusionPrior(
        net=prior_network,
        clip=OpenAIClipAdapter("ViT-L/14"),
        image_embed_dim=768,
        timesteps=1000,
        cond_drop_prob=0.1,
        loss_type="l2",
        condition_on_text_encodings=True,
    )
    
    state_dict = torch.load(PRIOR_PATH, map_location='cpu')
    if 'ema_model' in state_dict:
        print('Loading EMA Model')
        msg = diffusion_prior.load_state_dict(state_dict['ema_model'], strict=True)
        print("="*80)
        print(msg)
        print("="*80)
    else:
        print('Loading Standard Model')
        diffusion_prior.load_state_dict(state_dict['model'], strict=False)
        print("="*80)
        print(msg)
        print("="*80)
    del state_dict
    return diffusion_prior