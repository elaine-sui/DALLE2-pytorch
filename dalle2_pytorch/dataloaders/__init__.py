from dalle2_pytorch.dataloaders.decoder_loader import ImageEmbeddingDataset, create_image_embedding_dataloader
from dalle2_pytorch.dataloaders.prior_loader import make_splits, get_reader, PriorEmbeddingDataset

from .data_module import DataModule
from .dataset_coco import CocoDataset
ALL_DATASETS = {'coco': CocoDataset}