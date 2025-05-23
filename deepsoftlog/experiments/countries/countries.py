import torch

from deepsoftlog.experiments.countries.dataset import generate_prolog_files, get_test_dataloader, get_train_dataloader, get_val_dataloader
from deepsoftlog.experiments.countries.visualise import visualise_matrix, get_located_in_matrix, COUNTRIES
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

def visualise(trainer, logger, cfg):
    # visualise task
    region_matrix, subregion_matrix = get_located_in_matrix(cfg.name)
    fig1 = visualise_matrix(region_matrix, COUNTRIES)
    fig2 = visualise_matrix(subregion_matrix, COUNTRIES)
    logger.log_fig(fig1, name="region matrix")
    logger.log_fig(fig2, name="subregion matrix")
    # visualise soft unification scores
    names, matrix = trainer.program.get_soft_unification_matrix()
    fig = visualise_matrix(matrix, names)
    logger.log_fig(fig, name="similarity matrix")

def train(cfg):
    cfg = load_config(cfg)
    generate_prolog_files()
    eval_dataloader = get_val_dataloader()
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
    visualise(trainer, logger, cfg)


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
    visualise(trainer, logger, cfg)


if __name__ == "__main__":
    train("deepsoftlog/experiments/countries/config.yaml")

