from torch.utils.data import DataLoader
import pytorch_lightning as pl

from .dataset_coco_cap import build_dataset
from .constants import BATCH_SIZE, NUM_WORKERS

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size):
        super().__init__()
        
        self.batch_size = batch_size

        self.dataset = build_dataset()

    def shared_dataloader(self):
        return DataLoader(
            self.dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
        )
        
    def train_dataloader(self):
        return self.shared_dataloader()

    def val_dataloader(self):
        return self.shared_dataloader()

    def test_dataloader(self):
        return self.shared_dataloader()
    
    def predict_dataloader(self):
        return self.shared_dataloader()
    
    def all_dataloader(self):
        return self.shared_dataloader()