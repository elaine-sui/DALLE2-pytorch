PRIOR_PATH = "/pasteur/u/esui/data/dalle2/prior_aes_finetune.pth"
DATA_PATH="/pasteur/u/esui/data/coco/oscar_split_ViT-L_14_train_mini.pkl"
OUTPUT_EMBED_PATH = "/pasteur/u/esui/data/dalle2/coco_prior_output_ViT-L_14_train_mini.pkl"
L2_DISTANCE_FROM_AVG_SAMPLED = "/pasteur/u/esui/data/dalle2/coco_prior_output_ViT-L_14_train_mini_distance_from_avg.pkl"
L2_DISTANCE_BETWEEN_INPUT_OUTPUT = "/pasteur/u/esui/data/dalle2/coco_prior_output_ViT-L_14_train_mini_distance_btwn_input_output.pkl"
# prior_path = "/pasteur/u/esui/data/dalle2/diffusion_prior_vit-l-14-laion2b-ema855M.pth"

PRIOR_COND_SCALE=1.0
BATCH_SIZE=16
NUM_WORKERS=0
NUM_SAMPLES=100
NUM_REPEATS=50

AVG_OUTPUT_EMBED_FOLDER = f"/pasteur/u/esui/data/dalle2/coco_prior_output_ViT-L_14_train_mini_avg_{NUM_SAMPLES}x{NUM_REPEATS}"

COMBINED_PATH = AVG_OUTPUT_EMBED_FOLDER + '/prior_output_data.pkl'