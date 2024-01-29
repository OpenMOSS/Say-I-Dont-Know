import copy
import json

import torch
from torch.utils.data import Dataset

B_INST, E_INST = "[INST]", "[/INST]"

class QADataset(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        if partition == 'train':
            self.ann = json.load(open(dataset_config.train_data_path))
        else:
            self.ann = json.load(open(dataset_config.valid_data_path))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        IGNORE_INDEX = -100  # The default setting in CrossEntropyLoss


        ann = self.ann[index]
        prompt = B_INST + ' ' + ann['question'] + ' ' + E_INST
        example = prompt + ann["answer"]
        prompt = torch.tensor(
            self.tokenizer.encode(prompt), dtype=torch.int64
        )
        example = self.tokenizer.encode(example)
        example.append(self.tokenizer.eos_token_id)
        example = torch.tensor(
            example, dtype=torch.int64
        )
        labels = copy.deepcopy(example)
        labels[: len(prompt)] = -1
        example_mask = example.ge(0)
        label_mask = labels.ge(0)
        example[~example_mask] = 0
        labels[~label_mask] = IGNORE_INDEX

        return {
            "input_ids": example.tolist(),
            "labels": labels.tolist(),
            "attention_mask":example_mask.tolist(),
        }
