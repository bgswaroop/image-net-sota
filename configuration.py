import os
from pathlib import Path

# Set up the directories for the current training / evaluation session
root_session_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # This is the project root
# session_dir = Path("None")  # setting SESSION_DIR to "" (empty string), will create a new session directory
session_dir = root_session_dir.joinpath(r"runtime_data/2020-03-07_00.27.34/")
# session_dir = root_session_dir.joinpath(r"runtime_data/debug/")

# Data source
# root_data_dir = Path(r"D:\Data\imagenet-object-localization-challenge")
root_data_dir = Path(r"/data/p288722/imagenet-object-localization-challenge")

# Training configuration

