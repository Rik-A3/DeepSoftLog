import os

import torch

from deepsoftlog.experiments.mentions_countries.dataset import generate_prolog_files, get_val_dataloader, \
    get_train_dataloader, get_test_dataloader
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger, PrintLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

def train(cfg_path, name, seed, program, text_embedding_mode,
          device='cpu', device_nb=0):
    cfg = load_config(cfg_path)

    cfg.update({
        "name": name,
        "program": program,
        "seed": int(seed),
        "device": device,
        "device_nb": int(device_nb),
        "text_embedding_mode": text_embedding_mode,
    })
    _train(cfg)

def _train(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.device_nb)
    generate_prolog_files()
    eval_dataloader = get_val_dataloader()
    test_dataloader = get_test_dataloader()
    program = load_program(cfg, [get_train_dataloader(cfg), eval_dataloader, test_dataloader])
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)
    #logger = PrintLogger()
    trainer = Trainer(
        program, get_train_dataloader, nll_loss, optimizer,
        logger=logger,
        max_proofs=cfg['max_proofs'],
        max_branching=cfg['max_branching'],
        max_depth=cfg['max_depth'],
    )
    trainer.val_dataloader = eval_dataloader
    trainer.train(cfg)
    trainer.eval(test_dataloader)
    logger.finish()


def eval(folder: str):
    cfg = load_config(f"results/{folder}/config.yaml")
    eval_dataloader = get_test_dataloader()
    program = load_program(cfg, eval_dataloader)
    state_dict = torch.load(f"results/{folder}/store.pt")
    program.store.load_state_dict(state_dict, strict=False)
    trainer = Trainer(program, None, None, None)
    trainer.max_branching = cfg['max_branching']
    trainer.max_depth = cfg['max_depth']
    trainer.eval(eval_dataloader)


if __name__ == "__main__":
    train("deepsoftlog/experiments/mentions_countries/config.yaml", 'test', 0, 'deepsoftlog/experiments/mentions_countries/data/tmp/countries_S1_relation2text.pl', 'LM')