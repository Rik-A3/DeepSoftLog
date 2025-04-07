import itertools
from collections.abc import Iterator
from pathlib import Path
import os
import random
from typing import Union, List

import numpy as np
import torch
import yaml

from deepsoftlog.embeddings.embedding_store import create_embedding_store
from ..parser.parser import parse_file


def set_seed(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class ConfigDict(dict):
    def __getattr__(self, name):
        if name in self:
            value = self[name]
            if isinstance(value, dict):
                return ConfigDict(value)
            return value
        raise AttributeError(f"ConfigDict has no attribute {name}")

    def save(self, file_name: str | Path):
        with open(file_name, "w+") as f:
            yaml.dump(dict(self), f)

    def copy(self):
        return ConfigDict(self)

def load_program(config, init_dataloader: Union["DataLoader", List["DataLoader"]]) -> "Program":
    config = load_config(config)
    set_seed(config['seed'])

    program = parse_file(
        config['program'],
        embedding_metric=config['embedding_metric'],
        semantics=config['semantics'],
    )
    init_dataloader = [init_dataloader] if not isinstance(init_dataloader, list) else init_dataloader
    vocab = [program] + [dataloader.dataset for dataloader in init_dataloader]
    program.store = create_embedding_store(config, vocab_sources=vocab)
    return program


def load_config(cfg) -> ConfigDict:
    if isinstance(cfg, dict):
        return ConfigDict(cfg)
    cfg = str(cfg)
    with open(cfg) as f:
        return ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
