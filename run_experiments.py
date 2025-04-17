import sys

from deepsoftlog.experiments.mnist_addition.addition import train as train_addition
from deepsoftlog.experiments.countries.countries import train as train_countries
from deepsoftlog.experiments.wiki_countries.wiki_countries import train as train_wiki_countries
from deepsoftlog.experiments.fsm.fsm import train as train_fsm
from deepsoftlog.experiments.mentions_countries.mentions_countries import train as train_mentions


def main(experiment_name, config_file):
    train_functions = {'mnist_addition': train_addition, 'countries': train_countries, 'wiki_countries': train_wiki_countries, 'fsm': train_fsm, "mentions_countries": train_mentions}
    assert experiment_name in train_functions.keys(), f"Experiment name must be one of {tuple(train_functions.keys())}"
    return train_functions[experiment_name](config_file)


if __name__ == "__main__":
    main(*sys.argv[1:])
    #main("wiki_countries", "deepsoftlog/experiments/wiki_countries/config_S1.yaml")