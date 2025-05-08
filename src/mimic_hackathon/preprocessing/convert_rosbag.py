import collections
import shutil
from pathlib import Path
from typing import Any, Callable

import einops
import numpy as np
import PIL
import zarr
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore
from torchvision import transforms

from mimic_hackathon.preprocessing.ros_to_np import convert_ros_msg_to_np


LABEL_FILE_NAME = "label.txt"
NAME_MAP_ROS_MIMIC = {
    "/faive/policy_output": "robot0_hand_joints_lowdim",
    "/franka/commanded_pose": "robot0_eef_pose_ref_lowdim",
    "/franka/proprioceptive_pose": "robot0_eef_pose_lowdim",
    "/cameras/wrist_top": "robot0_ego_wrist_top_rgb",
    "/cameras/wrist_bottom": "robot0_ego_wrist_bottom_rgb",
    "/cameras/fixed_0": "workspace_rgb",
}


def _process_images(images: np.ndarray) -> np.ndarray:
    # BGR -> RGB
    images = images[..., ::-1]

    # Resize to (224, 224)
    images = np.stack(
        [
            transforms.Resize((224, 224))(
                PIL.Image.fromarray(image, "RGB"),
            )
            for image in images
        ],
        axis=0,
    )
    # Columns first
    images = einops.rearrange(images, "t h w c -> t c h w")
    # Normalize
    return images.astype(np.float32) / 255.0


def _dict_apply(
    x: dict[str, Any],
    func: Callable,
) -> dict[str, Any]:
    result = {}
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = _dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def _read_rosbag(bag_file_path: Path) -> dict:
    """Reads a rosbag file and converts it to a dictionarry of topic arrays."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)

    def get_default_topic_store():
        return {"timestamp": [], "message": []}

    topic_arrays = collections.defaultdict(get_default_topic_store)
    with Reader(bag_file_path) as reader:
        init_time = reader.start_time

        connections = [x for x in reader.connections if x.topic in NAME_MAP_ROS_MIMIC]
        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
            msg = convert_ros_msg_to_np(msg, connection.msgtype)

            timestamp = timestamp - init_time

            topic = NAME_MAP_ROS_MIMIC[connection.topic]

            if topic.endswith("_rgb"):
                msg = _process_images(msg)

            topic_arrays[topic]["timestamp"].append(timestamp)
            topic_arrays[topic]["message"].append(msg)

    return _dict_apply(topic_arrays, np.stack)


def _convert_rosbag(
    *,
    topic_arrays: dict,
    target_episode_path: Path,
    label_path: Path,
) -> None:
    """Build a dataset compliant with Dexformer and ACT.

    Requirements:
    - The dictionary names must NOT contain any / characters, as this is used to create
    the file structure.
    - Data must be stored in a directory with the name of the task.
    - Inside the directory, each episode is stored in a separate file.
    - The files are named 'episode_{episode_idx}.ext'.

    Notes:
    - Topics are not synced, they are stored with their original timestamps.
    """
    with zarr.open(target_episode_path, "w") as root:
        root.attrs["sim"] = False

        for topic, data in topic_arrays.items():
            timestamps = np.round(data["timestamp"]).astype(np.uint64)

            if np.diff(timestamps).min() < np.diff(timestamps).mean() / 2:
                print(
                    f"episode '{target_episode_path.name}' has (a) hiccup(s) on topic"
                    f" '{topic}'."
                )

            values = data["message"]
            root.create_dataset(
                topic,
                shape=values.shape,
                dtype=values.dtype,
                chunks=(1, *values.shape[1:]),
                compression="gzip",
                compression_opts=9,
            )
            root[f"/{topic}"][...] = values

            root.create_dataset(
                f"{topic}_timestamps",
                shape=(len(timestamps),),
                dtype="uint64",
                chunks=(1,),
                compression="gzip",
                compression_opts=9,
            )
            root[f"/{topic}_timestamps"][...] = timestamps

    if label_path.exists():
        shutil.copy(label_path, Path(target_episode_path, LABEL_FILE_NAME))


def convert_rosbag(
    episode_name: str,
    *,
    data_dir: Path,
    out_dir: Path,
):
    """Converts a rosbag to the specified target dataset format."""

    raw_episode_path = Path(data_dir, episode_name)
    target_episode_path = out_dir / f"{episode_name}.zarr"

    try:
        topic_arrays = _read_rosbag(raw_episode_path)
    except Exception as e:
        print(f"Error reading rosbag: {e}")
        return 0

    _convert_rosbag(
        topic_arrays=topic_arrays,
        target_episode_path=target_episode_path,
        label_path=raw_episode_path / LABEL_FILE_NAME,
    )

    length = min(len(val["message"]) for val in topic_arrays.values())
    return length
