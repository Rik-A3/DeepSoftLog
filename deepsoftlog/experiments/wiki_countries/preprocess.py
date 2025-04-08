import os
from pathlib import Path

from transformers import AutoTokenizer
import torch

from deepsoftlog.data import load_tsv_file

ROOT = Path(__file__).parent

CONTEXT_SIZE = 128
def load_text_data():
	base_path = Path(__file__).parent / 'data' / 'raw'
	data = load_tsv_file(base_path / f"text.tsv")
	return [a[1] for a in data]

def load_masked_text_data():
	base_path = Path(__file__).parent / 'data' / 'raw'
	data = load_tsv_file(base_path / f"text.tsv")
	return [a[1] for a in data]

def make_token_list(tokenizer, text_data, context_size=CONTEXT_SIZE):
	token_dict = tokenizer(text_data)
	input_list = token_dict['input_ids']
	attention_mask_list = token_dict['attention_mask']

	tokens_tensor_list = [torch.tensor(x) for x in zip(input_list, attention_mask_list)]
	tokens_tensor_list = [torch.cat((x[:, 0:context_size],x[:, -1:]), dim=1) for x in tokens_tensor_list] # truncate to CONTEXT SIZE but keep eos token

	return tokens_tensor_list

def make_tensor_lists(root=ROOT, context_size=CONTEXT_SIZE):
	if not os.path.exists('data/raw/tokens_tensor_list.pt') or not os.path.exists('data/raw/masked_tokens_tensor_list.pt'):
		text_data = load_text_data()
		masked_text_data = load_masked_text_data()

		tokenizer = (AutoTokenizer.from_pretrained('roberta-base'))

		torch.save(make_token_list(tokenizer, text_data, context_size), root / 'data/raw/tokens_tensor_list.pt')
		torch.save(make_token_list(tokenizer, masked_text_data, context_size), root / 'data/raw/masked_tokens_tensor_list.pt')