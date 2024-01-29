import copy
import json

import torch
from torch.utils.data import Dataset

B_INST, E_INST = "[INST]", "[/INST]"

class RewardDatasetPPO(Dataset):
    def __init__(self, dataset_config, tokenizer, partition="train"):
        self.ann = json.load(open(dataset_config.data_path))
        if partition == 'train':
            self.ann = self.ann['train']
        else:
            self.ann = self.ann['test']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]
        chosen_token = B_INST + ' ' + ann['question'] + ' ' + E_INST + ' ' + ann['positive_answer']
        rejected_token = B_INST + ' ' + ann['question'] + ' ' + E_INST + ' ' + ann['negative_answer']

        return {
            "chosen_token": chosen_token,
            "rejected_token": rejected_token,
        }
