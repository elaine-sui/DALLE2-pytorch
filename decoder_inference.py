import torch
from omegaconf import OmegaConf
from dalle2_pytorch.builder import build_data_module
from dalle2_pytorch.train_configs import TrainDecoderConfig
import numpy as np
import os
from PIL import Image
import argparse
from torchvision.transforms import Resize, ToPILImage, Compose
from x_clip import CLIP
import clip
from dalle2_pytorch.tokenizer import tokenizer
import re

def conditioned_on_text(config):
    try:
        return config.decoder.unets[0].cond_on_text_encodings
    except AttributeError:
        pass
    
    try:
        return config.decoder.condition_on_text_encodings
    except AttributeError:
        pass
    
    return False

decoder_text_conditioned = False
clip_config = None
device = "cuda" if torch.cuda.is_available() else "cpu"
decoder_cond_scale = 3.5

def load_decoder(decoder_state_dict_path, config_file_path):
    config = TrainDecoderConfig.from_json_path(config_file_path)
    global decoder_text_conditioned
    config.decoder.unets[0].cond_on_text_encodings = True # False
    global clip_config
    clip_config = config.decoder.clip
    config.decoder.clip = None
    decoder_text_conditioned = conditioned_on_text(config)
    print("Decoder conditioned on text", decoder_text_conditioned)
    decoder = config.decoder.create().to(device)
    decoder_state_dict = torch.load(decoder_state_dict_path, map_location='cpu')
    
    # remap some decoder_state_dict keys
    decoder_state_dict = {re.sub(r'(\d).(\d).([a-z]+)', r'\1.\2.1.\3', k):v for k,v in decoder_state_dict.items()}
    
    msg = decoder.load_state_dict(decoder_state_dict, strict=False)
    print("="*80)
    print(msg)
    print("="*80)
    del decoder_state_dict
    decoder.eval()
    return decoder

def load_clip():
    if clip_config is not None:
        clip = clip_config.create()
    return clip

def load_dataloader(config, split):
    data_module = build_data_module(config)
    if split == "train":
        dataloader = data_module.train_dataloader()
    elif split == "train_sampling": 
        dataloader = data_module.train_dataloader(sampling=True)
    elif split == "val":
        dataloader = data_module.val_dataloader()
    elif split == "test": 
        dataloader = data_module.test_dataloader()
    elif split == "test_sampling": 
        dataloader = data_module.test_dataloader(sampling=True)
    else:
        raise NotImplementedError(f"invalid split {split}")
    
    return dataloader

def save_images(output_dir, np_images):
    os.makedirs(output_dir, exist_ok=True)
    for i, np_img in enumerate(np_images):
        image = Image.fromarray(np.uint8(np_img * 255))
        output_path = os.path.join(output_dir, f'{i}.png')
        image.save(output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_modality", type=str, default="language", choices=("language", "vision"))
    parser.add_argument("--split", type=str, default="test", 
                        choices=("train", "val", "test"))
    parser.add_argument("--remove_modality_gap", action="store_true")
    parser.add_argument("--remove_mean", action="store_true")
    parser.add_argument("--normalize_embed", action="store_true")
    
    args = parser.parse_args()
    
    # Load decoder
    decoder_state_dict_path = "/pasteur/u/esui/data/dalle2/1.5B_laion2B_latest.pth"
    config_file_path = "old_configs/1.5B_laion2B_decoder_config.json"
    decoder = load_decoder(decoder_state_dict_path, config_file_path)
    
    # Load dataloader
    data_config_path = "configs/data_cfg.yaml"
    data_config = OmegaConf.load(data_config_path)
    data_config.encoder.modality = args.input_modality
    data_config.test_split = args.split
    data_config.data.remove_modality_gap = args.remove_modality_gap
    data_config.data.remove_mean = args.remove_mean
    data_config.data.normalize_embed = args.normalize_embed
    
    data_loader = load_dataloader(data_config, split=args.split)
    
    clip_x = load_clip()
    
    post_processing = Compose([ToPILImage(), Resize(224)])
    
    output_dir = f"output/images_from_{args.input_modality}"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (img, emb, txt) in enumerate(data_loader):
        # Note: single image
        print(f"Caption: {txt}")
        embeddings = emb.get('img').type(torch.FloatTensor).to(device)
        tokens = tokenizer.tokenize(txt).to(device)
        
        if decoder_text_conditioned:
            print("Generating clip embeddings")
            # _, text_encoding, text_mask = clip_x.embed_text(tokens)
            image = decoder.sample(embeddings, text=txt, cond_scale = decoder_cond_scale)
        else:
            image = decoder.sample(image_embed=embeddings, text = None, cond_scale = decoder_cond_scale)
        
        image = image.cpu().permute(0, 2, 3, 1).numpy().squeeze() * 255
        image = image.astype(np.uint8)
        
        image = post_processing(image)
        image = np.asarray(image)
        
        image = Image.fromarray(image)
        caption = txt[0].lower().replace(' ', '_').replace('.', '')
        print(caption)
        
        output_path = os.path.join(output_dir, f'{caption}.png')
        image.save(output_path)
        import pdb; pdb.set_trace()
        
        # with torch.no_grad():
        #     if decoder_text_conditioned:
        #         print("Generating clip embeddings")
        #         _, text_encoding, text_mask = clip.embed_text(tokens)
        #         images = decoder.sample(embeddings, text_encodings = text_encoding, text_mask = text_mask, cond_scale = decoder_cond_scale)
        #     else:
        #         print("Not generating clip embeddings")
        #         images = decoder.sample(embeddings, text = None, cond_scale = decoder_cond_scale)
        # images = images.cpu().permute(0, 2, 3, 1).numpy()
        # np.save('images_decoder.npy', images)