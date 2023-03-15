PRIOR_PATH = "./data/dalle2/prior_aes_finetune.pth"
DATA_PATH="./data/coco/oscar_split_ViT-L_14_train_mini.pkl"
OUTPUT_EMBED_PATH = "./data/dalle2/coco_prior_output_ViT-L_14_train_mini.pkl"
L2_DISTANCE_FROM_AVG_SAMPLED = "./data/dalle2/coco_prior_output_ViT-L_14_train_mini_distance_from_avg.pkl"
L2_DISTANCE_BETWEEN_INPUT_OUTPUT = "./data/dalle2/coco_prior_output_ViT-L_14_train_mini_distance_btwn_input_output.pkl"

PRIOR_COND_SCALE=1.0
BATCH_SIZE=16
NUM_WORKERS=0
NUM_SAMPLES=100
NUM_REPEATS=50

AVG_OUTPUT_EMBED_FOLDER = f"./data/dalle2/coco_prior_output_ViT-L_14_train_mini_avg_{NUM_SAMPLES}x{NUM_REPEATS}"

COMBINED_PATH = AVG_OUTPUT_EMBED_FOLDER + '/prior_output_data.pkl'