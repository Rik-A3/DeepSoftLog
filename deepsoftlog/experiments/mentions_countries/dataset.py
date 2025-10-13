import pandas as pd
import numpy as np

from pathlib import Path

from deepsoftlog.data import load_tsv_file, data_to_prolog
from deepsoftlog.algebraic_prover.terms.expression import Constant
from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.data.dataset import StaticDataset
from deepsoftlog.logic.soft_term import SoftTerm

_DATA_ROOT = str(Path(__file__).parent / "tmp")

def generate_tsvs(country_to_text: bool, relation_to_text: bool):
    base_path = Path(__file__).parent / 'data' / 'raw'

    country_entity_text = pd.read_csv(base_path / 'country2text.csv', dtype=str).set_index('Country')
    country2text = {e[0]: np.random.permutation(e[2:]) for e in country_entity_text.itertuples()}
    relation2text = {
        'locatedIn': ['is positioned in', 'was positioned in', 'can be found in', 'was located in', 'is located in',
                      'is situated in', 'was situated in', 'can be found in', 'was placed in', 'is placed in',
                      'is currently in', 'was currently in', 'was still in', 'is still in', 'is present in',
                      'was present in', 'is localized in', 'was localized in', 'is sited in', 'was sited in'],
        'neighborOf': ['was a neighboring country of', 'is a neighboring country of', 'neighbors', 'is adjacent to',
                       'was adjacent to', 'was butted against', 'is butted against', 'borders', 'borders with',
                       'was a neighboring state to', 'is a neighboring state to', 'is a neighbor of',
                       'was a neighbor of', 'neighbors with']}

    for task in ["S1", "S2", "S3"]:
        # load symbolic facts
        symbolic_facts = load_tsv_file(base_path / f"countries_{task}.tsv")

        for i, fact in enumerate(symbolic_facts):
            if country2text:
                text_1 = "text({})".format(country2text[fact[0]][0].replace(" ", ":"))
                country2text[fact[0]] = np.roll(country2text[fact[0]], 1)
                text_2 = "text({})".format(country2text[fact[2]][0].replace(" ", ":"))
                country2text[fact[2]] = np.roll(country2text[fact[2]], 1)
            else:
                text_1 = fact[0]
                text_2 = fact[2]

            if relation2text:
                text_r = "text({})".format(relation2text[fact[1]][0].replace(" ", ":"))
                relation2text[fact[1]] = np.roll(relation2text[fact[1]], 1)
            else:
                text_r = fact[1]

            symbolic_facts[i] = (text_1, text_r, text_2)

        with open(base_path / Path(f"countries_{task}" + ('_country2text' if country_to_text else '') + (
        '_relation2text' if relation_to_text else '') + ".tsv"), "w") as f:
            f.write("\n".join(["\t".join(fact) for fact in symbolic_facts]))


def get_train_dataloader(cfg: dict):
    train_dataset = MentionsCountriesDataset("train").subset(cfg['data_subset'])
    train_dataset = train_dataset.mutate_all_output()
    return DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, seed=cfg['seed'])


def get_test_dataloader():
    regions = ["africa", "americas", "asia", "europe", "oceania"]
    domain = {-1: [SoftTerm(Constant(r)) for r in regions]}
    eval_dataset = MentionsCountriesDataset("test").mutate_all_output(domain)
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)

def get_val_dataloader():
    regions = ["africa", "americas", "asia", "europe", "oceania"]
    domain = {-1: [SoftTerm(Constant(r)) for r in regions]}
    eval_dataset = MentionsCountriesDataset("val").mutate_all_output(domain)
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)

class MentionsCountriesDataset(StaticDataset):
    def __init__(self, split_name: str = "val"):
        base_path = Path(__file__).parent / 'data' / 'raw'
        data = load_tsv_file(base_path / f"{split_name}.tsv")
        data = data_to_prolog(data, name="countries")
        super().__init__(tuple(data))

def generate_prolog_files():
    base_path = Path(__file__).parent / 'data'
    (base_path / 'tmp').mkdir(exist_ok=True)
    for setting in ['', '_country2text', '_relation2text', '_country2text_relation2text']:
        for problem in (f'S{i}' for i in range(1,4)):
            data = load_tsv_file(base_path / f"raw/countries_{problem}{setting}.tsv")
            data = data_to_prolog(data, name="countries")
            file_str = [f"{query.query}." for query in data]
            # add template stuff
            with open(base_path / f"templates/countries_{problem}_templates.pl", "r") as f:
                templates = f.read()
            with open(base_path / f"tmp/countries_{problem}{setting}.pl", "w") as f:
                f.write("\n".join(file_str) + "\n")
                f.write(templates)

if __name__ == "__main__":
    d = MentionsCountriesDataset()
    print(d)
    generate_prolog_files()