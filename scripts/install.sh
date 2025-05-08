#!/bin/bash

# Install micromamba.
"${SHELL}" <(curl -LsS micro.mamba.pm/install.sh)
# Activate micromamba on current shell.
eval "$(micromamba shell hook --shell bash)"
# Create empty virtual environment.
micromamba create -n mimic_hackathon python=3.10.12 -y
# Install poetry.
curl -LsS https://install.python-poetry.org | python3 -
# Activate virtual environment.
micromamba activate mimic_hackathon
# Install dependencies.
poetry install
# Patch: Fix bug in rosbags/rosbag2/storage_sqlite3.py file.
VENV_PATH=$(micromamba env list | awk '/\*/ {print $NF}')
PYTHON_FILE=$VENV_PATH/lib/python3.10/site-packages/rosbags/rosbag2/storage_sqlite3.py
sed -i '/cur = conn.cursor()/,/raise ReaderError(msg)/d' $PYTHON_FILE
# Install git hook scripts.
pre-commit install
