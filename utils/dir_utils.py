import sys
import os
import pathlib
import logging

def create_dir(dir_path):
    try:
        if not os.path.exists(dir_path):
            pathlib.Path(dir_path).mkdir(parents=True, exist_ok=True)
        return 1
    except:
        return 0