import os
from pathlib import Path

# Set up the directories for the current training / evaluation session
root_session_dir = Path(os.path.dirname(os.path.abspath(__file__)))  # This is the project root
session_dir = Path("None")  # setting SESSION_DIR to "" (empty string), will create a new session directory

# Data source
# root_data_dir = Path(r"D:\Data\imagenet-object-localization-challenge")
root_data_dir = Path(r"/data/p288722/imagenet-object-localization-challenge")

# Training configuration

