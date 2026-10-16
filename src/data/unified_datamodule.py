from typing import Any, List, Optional

from torch.utils.data import ConcatDataset, DataLoader


class UnifiedDataModule:
    """Concatenates the configured datasets and builds the train / test dataloaders.

    Test batches contain one human scanpath each (``batch_size_test=1``); the evaluator
    groups them by stimulus and generates one sample per human viewer.
    """

    def __init__(
        self,
        collator,
        train_datasets: Optional[List] = None,
        test_datasets: Optional[List] = None,
        batch_size_train: int = 128,
        batch_size_test: int = 1,
        num_workers: int = 8,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.collator = collator
        self.train_datasets = train_datasets or []
        self.test_datasets = test_datasets or []
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last

    def train_dataloader(self) -> DataLoader[Any]:
        assert self.train_datasets, "No training datasets configured."
        return DataLoader(
            dataset=ConcatDataset(self.train_datasets),
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            collate_fn=self.collator,
            drop_last=self.drop_last,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        assert self.test_datasets, "No test datasets configured."
        return DataLoader(
            dataset=ConcatDataset(self.test_datasets),
            batch_size=self.batch_size_test,
            pin_memory=self.pin_memory,
            shuffle=False,
            collate_fn=self.collator,
            drop_last=self.drop_last,
        )
