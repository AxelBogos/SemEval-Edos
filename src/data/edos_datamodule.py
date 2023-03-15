from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from src.data.components.Dataset import GenericDataset
from src.data.file_preprocessing import FilePreprocessor
from src.data.text_processing import SpacyTokenizer, TextPreprocessor
from src.main_scripts.downloader import GoogleDriveDownloader


class EDOSDataModule(LightningDataModule):
    """A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = Path("..", "..", "data", "edos_raw").resolve().as_posix(),
        preprocessing_mode: str = "standard",
        task: str = "a",
        max_length: int = 512,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        # data preparation handlers
        self.download_handler = GoogleDriveDownloader(output_dir=self.hparams.data_dir)
        self.text_preprocessor = TextPreprocessor(
            preprocessing_mode=self.hparams.preprocessing_mode
        )
        self.tokenizer = SpacyTokenizer()
        self.file_processor = FilePreprocessor(data_root=self.hparams.data_dir)

    def prepare_data(self):
        self.download_handler.download()
        self.file_processor.run()

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """

        if not self.data_train:
            train_path = Path(self.hparams.data_dir, "train_all_tasks_target_encoded.csv")
            raw_data_train = pd.read_csv(train_path).to_numpy()
            self.data_train = GenericDataset(
                raw_data_train[:, [1, self._train_target_index]],
                self.tokenizer,
                self.text_preprocessor,
                self.hparams.max_length,
            )
        if not self.data_val:
            val_path = Path(self.hparams.data_dir, f"dev_task_{self.hparams.task}_labelled.csv")
            raw_data_val = pd.read_csv(val_path).to_numpy()
            self.data_val = GenericDataset(
                raw_data_val[:, [1, 2]],
                self.tokenizer,
                self.text_preprocessor,
                self.hparams.max_length,
            )
        if not self.data_test:
            test_path = Path(self.hparams.data_dir, f"test_task_{self.hparams.task}_labelled.csv")
            raw_data_test = pd.read_csv(test_path).to_numpy()
            self.data_test = GenericDataset(
                raw_data_test[:, [1, 2]],
                self.tokenizer,
                self.text_preprocessor,
                self.hparams.max_length,
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass

    @property
    def _num_classes(self):
        if self.hparams.task == "a":
            return 2
        elif self.hparams.task == "b":
            return 4
        elif self.hparams.task == "c":
            return 11

    @property
    def _train_target_index(self):
        if self.hparams.task == "a":
            return 2
        elif self.hparams.task == "b":
            return 3
        elif self.hparams.task == "c":
            return 4


if __name__ == "__main__":
    _ = EDOSDataModule()
