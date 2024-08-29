import lightning as L
from torch.utils.data import DataLoader
from litdata import StreamingDataset, StreamingDataLoader, TokensLoader, train_test_split, CombinedStreamingDataset
import torch
import os

from datasets import load_dataset


class TextDataModule(L.LightningDataModule):
    def __init__(
        self, data_path, max_seq_len, num_val_samples, train_batch_size, 
        val_batch_size, *args, num_proc=1, shuffle=True, **kwargs):

        super().__init__()
        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

        self._train_dataloader = None
        self._val_dataloader = None

        self.max_seq_len = max_seq_len

        self.val_size = num_val_samples
        self.shuffle = shuffle

        self.num_proc = num_proc

        self.additional_args = args
        self.additional_kwargs = kwargs

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            path_subdirs = os.listdir(self.data_path)
            if "index.json" not in path_subdirs:
                data_paths = [os.path.join(self.data_path, subdir) for subdir in path_subdirs]
                try:
                    subdir_datasets = [
                        StreamingDataset(
                            input_dir=sub_data_path,
                            item_loader=TokensLoader(block_size=self.max_seq_len + 1),
                            shuffle=self.shuffle,
                            drop_last=True,
                        ) for sub_data_path in data_paths
                    ]
                except ValueError:
                    raise ValueError(f"{self.data_path} should either contain litdata processed files or subfolders containing litdata processed files.")
                
                subdir_datasets_split = [
                    train_test_split(ds, [1.-self.val_size/len(ds), self.val_size/len(ds)]) for ds in subdir_datasets
                ]
                
                self.train_dataset = CombinedStreamingDataset(
                    datasets=[ds[0] for ds in subdir_datasets_split], iterate_over_all=False
                )
                self.val_dataset = CombinedStreamingDataset(
                    datasets=[ds[1] for ds in subdir_datasets_split], iterate_over_all=False
                )
            else:
                dataset = StreamingDataset(
                    input_dir=self.data_path,
                    item_loader=TokensLoader(block_size=self.max_seq_len + 1),
                    shuffle=self.shuffle,
                    drop_last=True,
                )
                val_ratio = self.val_size/len(dataset)
                self.train_dataset, self.val_dataset = train_test_split(dataset, [1.-val_ratio, val_ratio])
            

    def train_dataloader(self):
        if self._train_dataloader is None:
            self._train_dataloader = StreamingDataLoader(
                self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_proc, pin_memory=True
            )
        return self._train_dataloader

    def val_dataloader(self):
        if self._val_dataloader is None:
            self._val_dataloader = StreamingDataLoader(
                self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_proc, pin_memory=True
            )
        return self._val_dataloader
    
    def state_dict(self):
        # track whatever you want here
        state = {"train_dl_state_dict": self._train_dataloader.state_dict()}
        return state

    def load_state_dict(self, state_dict):
        # restore the state based on what you tracked in (def state_dict)
        self.train_dataloader().load_state_dict(state_dict["train_dl_state_dict"])