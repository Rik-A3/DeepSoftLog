from pathlib import Path
import random

import torch
from torch.utils.data import Dataset as TorchDataset

from deepsoftlog.data import load_tsv_file, data_to_prolog, Query, to_prolog_text, load_txt_file, load_lines
from deepsoftlog.algebraic_prover.terms.expression import Constant, Expr
from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.data.dataset import StaticDataset, Dataset
from deepsoftlog.logic.soft_term import SoftTerm

_DATA_ROOT = str(Path(__file__).parent / "tmp")

class WikiDataset(TorchDataset):
    def __init__(self, masked=False):
        base_path = Path(__file__).parent / 'data' / 'raw'
        self.data = torch.load(base_path / ("masked_" if masked else "").__add__("tokens_tensor_list.pt"))

        super().__init__()

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class CountriesDataset(StaticDataset):
    def __init__(self, split_name: str = "val"):
        base_path = Path(__file__).parent / 'data' / 'raw'
        data = load_tsv_file(base_path / f"wikicountries_{split_name}.tsv")
        super().__init__(tuple(data))

class CountriesOperator(Dataset, TorchDataset):
    def __init__(
        self,
        split_name: str,
        function_name: str,
        masked=False,
        seed=None,
    ):
        """Generic data for operator(img, img) style datasets.
        :param split_name: Dataset to use (training, val, test)
        :param function_name: Name of Problog function to query.
        :param operator: Operator to generate correct examples
        :param size: Size of numbers (number of digits)
        :param arity: Number of arguments for the operator
        :param seed: Seed for RNG
        """
        super(CountriesOperator, self).__init__()
        self.split_name = split_name
        self.text_dataset = WikiDataset(masked)
        self.countries_dataset = CountriesDataset(split_name)
        self.function_name = function_name
        self.seed = seed

        indices = list(range(len(self.countries_dataset)))
        if seed is not None:
            rng = random.Random(seed)
            rng.shuffle(indices)

        self.data_indices = indices

    def __getitem__(self, i: int) -> Query:
        """Generate queries"""
        country_fact = self.countries_dataset[self.data_indices[i]]

        relation = SoftTerm(Constant(country_fact[1]))
        first_entity = to_prolog_text(self.text_dataset[int(country_fact[0])]) if country_fact[0].isdigit() else SoftTerm(Constant(country_fact[0]))
        second_entity = to_prolog_text(self.text_dataset[int(country_fact[2])]) if country_fact[2].isdigit() else SoftTerm(Constant(country_fact[2]))

        return Query(Expr(self.function_name, relation, first_entity, second_entity))

    def __len__(self):
        return len(self.data_indices)

def get_train_dataloader(cfg):
    dataset = CountriesOperator(
        split_name="train",
        function_name="countries",
        masked=cfg.masked,
        seed=cfg.seed
    )
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

def get_val_dataloader(cfg):
    dataset = CountriesOperator(
        split_name="val",
        function_name="countries",
        masked=cfg.masked,
        seed=cfg.seed
    )
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

def get_test_dataloader(cfg):
    dataset = CountriesOperator(
        split_name="test",
        function_name="countries",
        masked=cfg.masked,
        seed=cfg.seed
    )
    return DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)

def generate_prolog_files():
    base_path = Path(__file__).parent / 'data'
    (base_path / 'tmp').mkdir(exist_ok=True)
    for problem in (f'S{i}' for i in range(4)):
        data = load_tsv_file(base_path / f"raw/countries_{problem}.tsv")
        data = data_to_prolog(data, name="countries")
        file_str = [f"{query.query}." for query in data]
        with open(base_path / f"tmp/countries_{problem}.pl", "w+") as f:
            f.write("\n".join(file_str))

        # add template stuff
        with open(base_path / f"templates/countries_{problem}_templates.pl", "r") as f:
            templates = f.read()
        with open(base_path / f"tmp/countries_{problem}.pl", "a+") as f:
            f.write("\n" + templates)



if __name__ == "__main__":
    d = CountriesOperator("train", "countries")
    print(d)
    #generate_prolog_files()
