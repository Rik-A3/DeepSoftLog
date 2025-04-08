import os

import torch

from deepsoftlog.experiments.wiki_countries.dataset import generate_prolog_files, get_val_dataloader, \
    get_train_dataloader, get_test_dataloader
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

def train(cfg_path):
    cfg = load_config(cfg_path)
    if isinstance(cfg.seed, list):
        for seed in cfg.seed:
            cfg.seed = seed
            _train(cfg)
    else:
        _train(cfg)

def _train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_nb)
    generate_prolog_files()
    eval_dataloader = get_val_dataloader(cfg)
    program = load_program(cfg, eval_dataloader)
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)
    trainer = Trainer(
        program, get_train_dataloader, nll_loss, optimizer,
        logger=logger,
        max_proofs=cfg['max_proofs'],
        max_branching=cfg['max_branching'],
        max_depth=cfg['max_depth'],
    )
    trainer.val_dataloader = eval_dataloader
    trainer.train(cfg)
    trainer.eval(get_test_dataloader())


def eval(folder: str):
    cfg = load_config(f"results/{folder}/config.yaml")
    eval_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, eval_dataloader)
    state_dict = torch.load(f"results/{folder}/store.pt")
    program.store.load_state_dict(state_dict, strict=False)
    trainer = Trainer(program, None, None, None)
    trainer.max_branching = cfg['max_branching']
    trainer.max_depth = cfg['max_depth']
    trainer.eval(eval_dataloader)


if __name__ == "__main__":
    train("deepsoftlog/experiments/wiki_countries/config_S0.yaml")