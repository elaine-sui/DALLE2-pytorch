import pickle
import random
import matplotlib.pyplot as plt 
import numpy as np
import os
from sklearn.manifold import TSNE

from enums import Modality
from test_diffusion_prior import OUTPUT_EMBED_PATH, DATA_PATH, AVG_OUTPUT_EMBED_PATH

def sample_embeds(data_path, key, n=100, normalize=True, to_numpy=True):
    with open(data_path, 'rb') as f:
        all_data = pickle.load(f)
        
    data = all_data[key]

    indices = random.sample(list(data.keys()), n)
    
    if to_numpy:
        embeds = [data[idx]["embed"].squeeze().numpy() for idx in indices]
    else:
        embeds = [data[idx]["embed"].squeeze() for idx in indices]
    embeds = np.stack(embeds)
    
    if normalize:
        embeds = embeds / np.linalg.norm(embeds, axis=1).reshape(-1, 1)
    
    return embeds


def plot_embeds(n=100):
    embeds_language_og = sample_embeds(DATA_PATH, key="captions", n=n)
    embeds_vision_og = sample_embeds(DATA_PATH, key="images", n=n)
    embeds_prior_out = sample_embeds(OUTPUT_EMBED_PATH, key="diffusion_prior_outputs", n=n, to_numpy=False)
    
    tsne = TSNE()
    
    embeds = np.vstack([embeds_language_og, embeds_vision_og, embeds_prior_out])
    
    two_dimensional_embeds = tsne.fit_transform(embeds)
    
    plt.clf()
    plt.figure()
    plt.scatter(two_dimensional_embeds[:n, 0], two_dimensional_embeds[:n, 1], color='red', label="text")
    plt.scatter(two_dimensional_embeds[n:2*n, 0], two_dimensional_embeds[n:2*n, 1], color='blue', label="image")
    plt.scatter(two_dimensional_embeds[2*n:3*n, 0], two_dimensional_embeds[2*n:3*n, 1], color='green', label="diffusion prior output")
    plt.legend()
    
    os.makedirs("output", exist_ok=True)
    plt.savefig(f"output/coco_embeds.png")
    
if __name__ == '__main__':
    plot_embeds(n=100)