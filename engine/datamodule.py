import lightning as L
from torch.utils.data import DataLoader
import torch

from datasets import load_dataset

def collate_fn(batch):
    return {k: torch.stack([torch.tensor(b[k]) for b in batch]) for k in batch[0].keys()}


class TextDataModule(L.LightningDataModule):
    def __init__(self, data_path, num_val_samples, train_batch_size, val_batch_size, *args, shuffle=True, **kwargs):
        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self.val_size = num_val_samples
        self.shuffle = shuffle

        self.additional_args = args
        self.additional_kwargs = kwargs

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            dataset = load_dataset(self.data_path, *self.additional_args, **self.additional_kwargs)
            dataset_split = dataset["train"].train_test_split(
                test_size=self.val_size)
            self.train_dataset, self.val_dataset = dataset_split["train"], dataset_split["test"]
        if self.shuffle:
            self.train_dataset = self.train_dataset.shuffle()
            

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=0, collate_fn=collate_fn, pin_memory=True)