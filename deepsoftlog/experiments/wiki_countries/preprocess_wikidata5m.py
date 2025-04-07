import re
from pathlib import Path

import torch
from transformers import AutoTokenizer

from deepsoftlog.data import load_tsv_file
from deepsoftlog.experiments.countries.visualise import COUNTRIES

KEYWORDS = ['history', 'city', 'capital', 'country', 'province', 'island']
SUBCONTINENTS = ['southern_asia', 'south-eastern_asia', 'eastern_asia', 'central_asia', 'western_asia', 'northern_africa', 'middle_africa', 'western_africa', 'eastern_africa', 'southern_africa', 'northern_europe', 'western_europe', 'central_europe', 'eastern_europe', 'southern_europe', 'caribbean', 'northern_americas', 'central_america', 'south_america', 'polynesia', 'australia_and_new_zealand', 'melanesia', 'micronesia']
CONTINENTS = ['africa', 'americas', 'asia', 'europe', 'oceania']

def get_regions():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"countries_S0.tsv")
    lst = [a[0] for a in data] + [a[2] for a in data]
    return set([l.lower().replace("_", " ") for l in lst])

def convert_name(name):
    return name.lower().replace("_", " ")

def get_countries():
    regions = get_regions()
    return {r for r in regions if r not in [c.lower().replace("_", " ") for c in SUBCONTINENTS]
                                         + [c.lower().replace("_", " ") for c in CONTINENTS]}

def get_entities():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"wikidata5m_entity.tsv")
    return {a[0]: [s.lower() for s in a[1:]] for a in data}

def get_text():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"wikidata5m_text.tsv")
    return {a[0]: a[1] for a in data}

def get_text2():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"text_all_country_entities.tsv")
    return {a[0]: a[1] for a in data}


def get_country_entities():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"country_entities3.tsv")
    return {a[0]: [s for s in a[1:]] for a in data}

def get_full_kg():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"wikidata5m_inductive_test.tsv") + load_tsv_file(base_path / f"wikidata5m_inductive_train.tsv") + load_tsv_file(base_path / f"wikidata5m_inductive_valid.tsv")
    return data

def get_all_country_entities():
    base_path = Path(__file__).parent / 'data' / 'raw'
    data = load_tsv_file(base_path / f"all_country_entities.tsv")
    return {a[0]: [s for s in a[1:]] for a in data}

def get_new_id(entity_dict, old_id):
    if old_id in entity_dict:
        return min(entity_dict[convert_name(old_id)])
    return old_id


def make_token_list(tokenizer, text_data):
    token_dict = tokenizer(text_data)
    input_list = token_dict['input_ids']
    attention_mask_list = token_dict['attention_mask']

    tokens_tensor_list = [torch.tensor(x) for x in zip(input_list, attention_mask_list)]
    tokens_tensor_list = [torch.cat((x[:, 0:CONTEXT_SIZE],x[:, -1:]), dim=1) for x in tokens_tensor_list] # truncate to CONTEXT SIZE but keep eos token

    return tokens_tensor_list

CONTEXT_SIZE = 128
def generate_datasets_split_entities(entity_dict, text_dict):
    base_path = Path(__file__).parent / 'data' / 'raw'

    train_relations = load_tsv_file(base_path / "train.tsv")
    test_relations = load_tsv_file(base_path / "val.tsv")
    val_relations = load_tsv_file(base_path / "test.tsv")

    countries_S0 = load_tsv_file(base_path / "countries_S0.tsv")
    countries_S1 = load_tsv_file(base_path / "countries_S1.tsv")
    countries_S2 = load_tsv_file(base_path / "countries_S2.tsv")
    countries_S3 = load_tsv_file(base_path / "countries_S3.tsv")

    train_q_relations = [(min(entity_dict[convert_name(a[0])]), a[1], a[2]) for a in train_relations]
    test_q_relations = [(min(entity_dict[convert_name(a[0])]), a[1], a[2]) for a in test_relations]
    val_q_relations = [(min(entity_dict[convert_name(a[0])]), a[1], a[2]) for a in val_relations]

    countries_S0_relations = [(get_new_id(entity_dict, a[0]), a[1], a[2]) for a in countries_S0]
    countries_S1_relations = [(get_new_id(entity_dict, a[0]), a[1], a[2]) for a in countries_S1]
    countries_S2_relations = [(get_new_id(entity_dict, a[0]), a[1], a[2]) for a in countries_S2]
    countries_S3_relations = [(get_new_id(entity_dict, a[0]), a[1], a[2]) for a in countries_S3]

    text_data = []
    for i,a in enumerate(train_q_relations):
        text_data.append(text_dict[a[0]])
        train_q_relations[i] = (i, a[1], a[2])
        for i,c in enumerate(countries_S0_relations):
            if c[0] == a[0]:
                countries_S0_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S1_relations):
            if c[0] == a[0]:
                countries_S1_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S2_relations):
            if c[0] == a[0]:
                countries_S2_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S3_relations):
            if c[0] == a[0]:
                countries_S3_relations[i] = (i, c[1], c[2])
    n = len(text_data)
    for i,a in enumerate(test_q_relations):
        text_data.append(text_dict[a[0]])
        test_q_relations[i] = (n+i, a[1], a[2])
        for i,c in enumerate(countries_S0_relations):
            if c[0] == a[0]:
                countries_S0_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S1_relations):
            if c[0] == a[0]:
                countries_S1_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S2_relations):
            if c[0] == a[0]:
                countries_S2_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S3_relations):
            if c[0] == a[0]:
                countries_S3_relations[i] = (i, c[1], c[2])
    n = len(text_data)
    for i,a in enumerate(val_q_relations):
        text_data.append(text_dict[a[0]])
        val_q_relations[i] = (n+i, a[1], a[2])
        for i,c in enumerate(countries_S0_relations):
            if c[0] == a[0]:
                countries_S0_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S1_relations):
            if c[0] == a[0]:
                countries_S1_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S2_relations):
            if c[0] == a[0]:
                countries_S2_relations[i] = (i, c[1], c[2])
        for i,c in enumerate(countries_S3_relations):
            if c[0] == a[0]:
                countries_S3_relations[i] = (i, c[1], c[2])

    with open("data/raw/wikicountries_train.tsv", "w+") as f:
        for a in train_q_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")
    with open("data/raw/wikicountries_test.tsv", "w+") as f:
        for a in test_q_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")
    with open("data/raw/wikicountries_val.tsv", "w+") as f:
        for a in val_q_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")
    with open("data/raw/wikicountries_S0.tsv", "w+") as f:
        for a in countries_S0_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")
    with open("data/raw/wikicountries_S1.tsv", "w+") as f:
        for a in countries_S1_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")
    with open("data/raw/wikicountries_S2.tsv", "w+") as f:
        for a in countries_S2_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")
    with open("data/raw/wikicountries_S3.tsv", "w+") as f:
        for a in countries_S3_relations:
            f.write(str(a[0]) + "\t" + a[1] + "\t" + a[2] + "\n")

    tokenizer = AutoTokenizer.from_pretrained('roberta-base')

    masked_text_data = text_data.copy()
    for i in range(len(masked_text_data)):
        for c in COUNTRIES:
            masked_text_data[i] = re.sub(c.replace("_", " "), "<mask>", masked_text_data[i], flags=re.IGNORECASE)


    torch.save(make_token_list(tokenizer, text_data), 'data/raw/tokens_tensor_list.pt')
    torch.save(make_token_list(tokenizer, masked_text_data), 'data/raw/masked_tokens_tensor_list.pt')


if __name__ == "__main__":
    all_country_entities = get_all_country_entities()
    text_dict = get_text()
    text_dict.update(get_text2())

    generate_datasets_split_entities(all_country_entities, text_dict)

    # country_entities = get_country_entities()
    #
    # kg = get_full_kg()
    #
    # captial_relations = [a for a in kg if a[1] == 'P1376']
    # history_relations = [a for a in kg if a[1] == 'P2184']
    # flag_relations = [a for a in kg if a[1] == 'P163']
    #
    # result_dict = {c : [] for c in country_entities}
    # for c in country_entities:
    #     result_dict[c] += [a[0] for a in captial_relations if a[2] in country_entities[c]] + [a[2] for a in captial_relations if a[0] in country_entities[c]]
    #     result_dict[c] += [a[0] for a in history_relations if a[2] in country_entities[c]] + [a[2] for a in history_relations if a[0] in country_entities[c]]
    #     result_dict[c] += [a[0] for a in flag_relations if a[2] in country_entities[c]] + [a[2] for a in flag_relations if a[0] in country_entities[c]]
    #
    # with open("data/raw/country_entities4.tsv", "w+") as f:
    #     for c in country_entities:
    #         f.write(c + "\t")
    #         for r in result_dict[c]:
    #             f.write(r + "\t")
    #         f.write("\n")

    # all_entities = sum(get_all_country_entities().values(), [])
    #
    # text = get_text()
    # text_new = {}
    #
    # for e in all_entities:
    #     text_new[e] = text[e]
    #
    # with open("data/raw/text_all_country_entities.tsv", "w+") as f:
    #     for e in all_entities:
    #         f.write(e + "\t")
    #         f.write(text_new[e] + "\n")