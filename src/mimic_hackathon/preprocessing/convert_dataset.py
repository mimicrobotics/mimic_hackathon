"""Usage:
python -m mimic_hackathon.preprocessing.convert_dataset \
    --dataset-path /path/to/dataset/folder
"""

from pathlib import Path

import tyro

from mimic_hackathon.preprocessing.convert_rosbag import convert_rosbag
from mimic_hackathon.preprocessing.dataset_converter import DatasetConverter


def main(
    dataset_path: Path,
    n_workers: int = 10,
):
    dataset_name = dataset_path.name
    converted_subdataset_path = dataset_path / "converted"

    if not converted_subdataset_path.exists():
        print(f"Converting {dataset_path.name} dataset ...")
        converter = DatasetConverter(
            dataset_path=dataset_path,
            n_workers=n_workers,
        )
        converter.convert(convert_rosbag)
    else:
        print(f"{dataset_name} converted directory found. Ignoring.")


if __name__ == "__main__":
    tyro.cli(main)
