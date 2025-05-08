# mimic_hackathon

Mimic data pre-processing and remote inference server utils for the hackathon.

## Install

1. Clone repository:

   ```bash
   git clone git@github.com:mimicrobotics/mimic_hackathon.git
   ```

1. Create a virtual environment and install the dependencies:

   ```bash
   scripts/install.sh
   ```

1. Activate virtual environment:

   ```bash
   micromamba activate mimic_hackathon
   ```

## Convert Raw Dataset

Raw dataset structure:

```
dataset_name
├── dataset_name_episode0_timestamp
│   ├── dataset_name_episode0_timestamp_0.db3
│   ├── fixed_0.mp4
│   ├── label.txt
│   ├── metadata.txt
│   ├── wrist_bottom.mp4
│   ├── wrist_top.mp4
├── dataset_name_episode1_timestamp
├── ...
└── dataset_name_episodeN_timestamp
```

Zarr dataset structure:

```
dataset_name/converted
├── dataset_name_episode0_timestamp.zarr
│   ├── action (Tl, 22)
│   ├── action_timestamps (Tl,)
│   ├── label.txt
│   └── observations
│       ├── images
│       │   ├── fixed_0_view_rgb (Ti, 480, 640, 3)
│       │   ├── fixed_0_view_rgb_timestamps (Ti,)
│       │   ├── wrist_bottom_view_rgb (Ti, 480, 640, 3)
│       │   ├── wrist_bottom_view_rgb_timestamps (Ti,)
│       │   ├── wrist_top_view_rgb (Ti, 480, 640, 3)
│       │   └── wrist_top_view_rgb_timestamps (Ti,)
│       ├── qpos (Tl, 7)
│       ├── qpos_sync_img_idx_fixed_0_view_rgb (Tl,)
│       └── qpos_timestamps (Tl,)
├── dataset_name_episode1_timestamp.zarr
├── ...
└── dataset_name_episodeN_timestamp.zarr
```

Where:
- `Tl` is the number of action and proprio observations.
- `Ti` is the number of image observations.

Convert a raw dataset from rosbags to Zarr format:

```bash
python -m mimic_hackathon.preprocessing.convert_dataset \
   --dataset-path /path/to/dataset/folder
```

## Serve Model

1. Launch the policy server:

   ```bash
   python policy_server.py \
      --ip <server-ip> \
      --port <server-port> \
      --checkpoint-path /path/to/checkpoint
   ```

1. On a browser, check that the server is up and running:

   ```bash
   http://<server-ip>:<server-port>/docs
   ```
