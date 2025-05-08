# utils.py

import json
import os
from .config import CONFIG_FILE


def check_embeddings_status():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                return config.get("embeddings_created", False)
        except:
            return False
    return False


def save_embeddings_status(status=True):
    with open(CONFIG_FILE, "w") as f:
        json.dump({"embeddings_created": status}, f)
