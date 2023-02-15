from torch.utils.data import DataLoader, ConcatDataset
from .. import builder
from .. import enums

splits = ['train', 'val', 'test', 'restval']


class DataModule:
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.dataset = builder.build_dataset(cfg)
        self.transform = builder.build_transformation(self.cfg)

    def train_dataloader(self, sampling=False):
        split = self.cfg.data.train_split
        dataset = self.dataset(self.cfg, split=split, transform=self.transform)
        
        batch_size = self.cfg.train.n_sample_images if sampling else self.cfg.data.batch_size

        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            batch_size=batch_size,
            num_workers=self.cfg.data.num_workers,
        )

    def val_dataloader(self):
        dataset = self.dataset(self.cfg, split="val", transform=self.transform)
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )

    def test_dataloader(self, sampling=False):
        if self.cfg.test_split == "all":
            datasets = [self.dataset(self.cfg, split=split, transform=self.transform) for split in splits]
            dataset = ConcatDataset(datasets)
        else:
            dataset = self.dataset(self.cfg, split=self.cfg.test_split, transform=self.transform)
            
        batch_size = self.cfg.train.n_sample_images if sampling else self.cfg.data.batch_size    
        
        return DataLoader(
            dataset,
            pin_memory=True,
            drop_last=False,
            shuffle=False,
            batch_size=batch_size,
            num_workers=self.cfg.data.num_workers,
        )
    
    def all_dataloader(self):
        datasets = [self.dataset(self.cfg, split=split, transform=self.transform) for split in splits]
        dataset = ConcatDataset(datasets)
        return DataLoader(
            dataset,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            batch_size=self.cfg.data.batch_size,
            num_workers=self.cfg.data.num_workers,
        )