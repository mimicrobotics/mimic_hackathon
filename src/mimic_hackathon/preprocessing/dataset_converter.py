import multiprocessing
from collections.abc import Callable
from pathlib import Path
from typing import Dict
from functools import partial
from tqdm import tqdm

import numpy as np


def get_raw_episode_folder_names(dataset_path: Path) -> list[str]:
    return [
        folder_path.name
        for folder_path in dataset_path.iterdir()
        if folder_path.is_dir() and not folder_path.name.startswith("converted")
    ]


def conv_fn_wrapper(x: Callable, kwargs: Dict):
    return x(**kwargs)


class DatasetConverter:
    """Dataset converter that converts each episode independently and in parallel."""

    def __init__(
        self,
        dataset_path: Path,
        *,
        n_workers: int,
    ) -> None:
        self.dataset_path = dataset_path
        self.out_dir = self.dataset_path / "converted"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.n_workers = n_workers

    def convert(self, convert_fn: Callable, **kwargs) -> None:
        file_names = get_raw_episode_folder_names(self.dataset_path)

        if not file_names:
            raise ValueError(
                f"{self.dataset_path} directory does not contain any episode folders.",
            )

        print(f"Converting {len(file_names)} episodes:")

        conversion_kwargs = [
            {
                "episode_name": file_name,
                "data_dir": self.dataset_path,
                "out_dir": self.out_dir,
                **kwargs,
            }
            for file_name in file_names
        ]

        lengths = []
        conv_fn_partial = partial(conv_fn_wrapper, convert_fn)
        with multiprocessing.Pool(self.n_workers) as pool:
            with tqdm(
                total=len(conversion_kwargs),
                desc="Converting episodes",
                dynamic_ncols=True,
                position=0,
                leave=True,
            ) as pbar:
                for length in pool.imap_unordered(conv_fn_partial, conversion_kwargs):
                    lengths.append(length)
                    pbar.update()

        max_length = max(lengths)
        min_length = min(lengths)
        mean_length = np.mean(lengths)

        print(f"Maximum episode length: {max_length}")
        print(f"Minimum episode length: {min_length}")
        print(f"Mean episode length: {mean_length}")

        print("Finished converting all episodes.")
