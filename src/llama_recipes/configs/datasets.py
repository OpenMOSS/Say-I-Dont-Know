# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
from dataclasses import dataclass


@dataclass
class Triviaqa_llama2_7b_chat_threshold_1_0:
    dataset: str = "KB_triviaqa_llama2_7b_chat"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = os.getcwd() + "/Idk_datasets/sft_data/llama-2-7b-chat/triviaqa_train_threshold_1.0_sft_data.json"
    valid_data_path: str = os.getcwd() + "/Idk_datasets/sft_data/llama-2-7b-chat/triviaqa_valid_threshold_1.0_sft_data.json"

@dataclass
class Triviaqa_llama2_7b_chat_threshold_1_0_half_data:
    dataset: str = "KB_triviaqa_llama2_7b_chat"
    train_split: str = "train"
    test_split: str = "val"
    train_data_path: str = os.getcwd() + "/Idk_datasets/sft_data/llama-2-7b-chat/triviaqa_train_threshold_1.0_half_sft_data.json"
    valid_data_path: str = os.getcwd() + "/Idk_datasets/sft_data/llama-2-7b-chat/triviaqa_valid_threshold_1.0_sft_data.json"

@dataclass
class Triviaqa_llama2_7b_chat_threshold_1_0_preference_data:
    dataset: str = "KB_triviaqa_llama2_7b_chat"
    train_split: str = "train"
    test_split: str = "val"
    data_path: str = os.getcwd() + "/Idk_datasets/preference_data/llama-2-7b-chat/triviaqa_train_and_valid_llama2_7b_chat_threshold_1.0_preference_pairs_for_ppo_reward.json"