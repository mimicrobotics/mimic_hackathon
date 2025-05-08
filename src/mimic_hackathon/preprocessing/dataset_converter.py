import multiprocessing
from collections.abc import Callable
from multiprocessing import Process, Queue
from pathlib import Path

import numpy as np


def get_raw_episode_folder_names(dataset_path: Path) -> list[str]:
    return [
        folder_path.name
        for folder_path in dataset_path.iterdir()
        if folder_path.is_dir() and not folder_path.name.startswith("converted")
    ]


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

        file_names_queue = Queue()
        for file_name in file_names:
            file_names_queue.put(file_name)

        with multiprocessing.Manager() as manager:
            len_dict = manager.dict()

            workers = []
            for _ in range(self.n_workers):
                process = Process(
                    target=convert_fn,
                    args=(file_names_queue, len_dict),
                    kwargs={
                        "data_dir": self.dataset_path,
                        "out_dir": self.out_dir,
                        **kwargs,
                    },
                )
                workers.append(process)
                process.start()

            for process in workers:
                process.join()

            lengths = len_dict.values()

        max_length = max(lengths)
        min_length = min(lengths)
        mean_length = np.mean(lengths)

        print(f"Maximum episode length: {max_length}")
        print(f"Minimum episode length: {min_length}")
        print(f"Mean episode length: {mean_length}")

        print("Finished converting all episodes.")
