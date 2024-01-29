# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import yaml
import torch

from transformers import LlamaTokenizer

from llama_recipes.inference.model_utils import load_llama_from_config, load_auto_model_from_config
from llama_recipes.utils.train_utils import RewardModel

# Get the current file's directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory
parent_directory = os.path.dirname(current_directory)

# Append the parent directory to sys.path
sys.path.append(parent_directory)
from model_checkpointing import load_sharded_model_single_gpu

def save_hf_format(model, tokenizer, output_dir):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    # tokenizer.save_vocabulary(output_dir)

def main(
    fsdp_checkpoint_path="", # Path to FSDP Sharded model checkpoints
    consolidated_model_path="", # Path to save the HF converted model checkpoints
    HF_model_path_or_name="", # Path/ name of the HF model that include config.json and tokenizer_config.json (e.g. meta-llama/Llama-2-7b-chat-hf)
    reward_model=False # If the model is a reward model
    ):
    
    try:
        file_name = 'train_params.yaml'
        # Combine the directory and file name to create the full path
        train_params_path = os.path.join(fsdp_checkpoint_path, file_name)
        # Open the file
        with open(train_params_path, 'r') as file:
            # Load the YAML data
            data = yaml.safe_load(file)

            # Access the 'model_name' field
            HF_model_path_or_name = data.get('model_name')

            print(f"Model name: {HF_model_path_or_name}")
    except FileNotFoundError:
        print(f"The file {train_params_path} does not exist.")
        if HF_model_path_or_name == "":
            HF_model_path_or_name = input("Please enter the model name: ")
        print(f"Model name: {HF_model_path_or_name}")
    except Exception as e:
        print(f"An error occurred: {e}")
        
        
    #load the HF model definition from config
    if reward_model:
        model_def = load_auto_model_from_config(HF_model_path_or_name)
    else:
        model_def = load_llama_from_config(HF_model_path_or_name)
    print("model is loaded from config")
    #loading the tokenizer form the  model_path
    tokenizer = LlamaTokenizer.from_pretrained(HF_model_path_or_name)
    #load the FSDP sharded checkpoints into the model
    if reward_model:
        model_def = RewardModel(base_model=model_def, tokenizer=tokenizer, num_padding_at_beginning=0, compute_fp32_loss=False, score_head_dtype=model_def.config.torch_dtype)
    model = load_sharded_model_single_gpu(model_def, fsdp_checkpoint_path)
    print("model is loaded from FSDP checkpoints")
    tokenizer.save_pretrained(consolidated_model_path)
    #save the FSDP sharded checkpoints in HF format
    if reward_model:
        save_hf_format(model, tokenizer, consolidated_model_path)
    else:
        model.save_pretrained(consolidated_model_path)
    print(f"HuggingFace model checkpoints has been saved in {consolidated_model_path}")
if __name__ == "__main__":
    fire.Fire(main)
