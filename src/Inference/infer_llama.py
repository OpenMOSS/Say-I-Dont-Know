# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time
import math
import json
import yaml

import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaConfig, LlamaForSequenceClassification
from peft import PeftModel
from tqdm import tqdm
from torch.nn.functional import softmax

B_INST, E_INST = "[INST]", "[/INST]"

def format_tokens_triviaqa(dialogs, tokenizer):
    batched_input = []
    for dialog in dialogs:
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.
        """
        batched_input.append(f"{B_INST} {(dialog).strip()} {E_INST}")
    
    batched_input = tokenizer(
        batched_input,
        return_tensors="pt",
        padding=True,
    )

    for k in batched_input:
        batched_input[k] = batched_input[k].cuda()
        
    return batched_input

def format_tokens_triviaqa_for_ppo_reward(dialogs, tokenizer, refuse_answer=None):
    batched_input = []
    for dialog in dialogs:
        """
        Please verify that your tokenizer support adding "[INST]", "[/INST]" to your inputs.
        Here, we are adding it manually.generated_answer
        """
        if refuse_answer is not None:
            batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {refuse_answer}")
            # batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {dialog['negative_answer']}")
        else:
            batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {dialog['generated_answer']}")
            # batched_input.append(f"{B_INST} {dialog['question']} {E_INST} {dialog['positive_answer']}")

    batched_input = tokenizer(
        batched_input,
        return_tensors="pt",
        padding=True,
    )

    for k in batched_input:
        batched_input[k] = batched_input[k].cuda()
        
    return batched_input

def read_dialogs_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        dialogs = json.load(file)
    return dialogs

# Function to load the main model for text generation
def load_model(model_name, quantization):
    model = LlamaForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        load_in_8bit=quantization,
        # device_map="auto",
        low_cpu_mem_usage=True,
    ).cuda()
    return model

def load_reward_model(model_name):
    model = LlamaForSequenceClassification.from_pretrained(model_name).cuda()
    return model

def load_dpo_model(model_name, quantization, best_dpo_model_on_dev=False):
    '''
    model_name is a dir path which contains a config.yaml file
    '''
    with open(os.path.join(model_name, 'config.yaml'), 'r', encoding='utf-8') as f:
        config = yaml.load(f.read(), Loader=yaml.FullLoader)
    model = LlamaForCausalLM.from_pretrained(
        config['model']['name_or_path'],
        return_dict=True,
        load_in_8bit=quantization,
        # device_map="auto",
        low_cpu_mem_usage=True,
    ).bfloat16()
    if best_dpo_model_on_dev:
        if os.path.exists('{}/step-BEST/policy.pt'.format(model_name)):
            model.load_state_dict(torch.load('{}/step-BEST/policy.pt'.format(model_name))['state'])
        else:
            raise ValueError('No BEST checkpoint found in {}'.format(model_name))
    else:
        model.load_state_dict(torch.load('{}/LATEST/policy.pt'.format(model_name))['state'])
    model = model.cuda()
    return model

# Function to load the PeftModel for performance optimization
def load_peft_model(model, peft_model):
    peft_model = PeftModel.from_pretrained(model, peft_model)
    return peft_model

# Loading the model from config to load FSDP checkpoints into that
def load_llama_from_config(config_path):
    model_config = LlamaConfig.from_pretrained(config_path) 
    model = LlamaForCausalLM(config=model_config)
    return model

def main(
    model_name,
    peft_model: str=None,
    batch_size: int=1,
    quantization: bool=False,
    max_new_tokens =512, #The maximum numbers of tokens to generate
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    do_sample: bool=True, #Whether or not to use sampling ; use greedy decoding otherwise.
    use_cache: bool=True,  #[optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
    top_p: float=0.9, # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    temperature: float=1.0, # [optional] The value used to modulate the next token probabilities.
    top_k: int=None, # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
    repetition_penalty: float=1.0, #The parameter for repetition penalty. 1.0 means no penalty.
    length_penalty: int=1, #[optional] Exponential penalty to the length that is used with beam-based generation.
    use_fast_kernels: bool = False, # Enable using SDPA from PyTorch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
    response_num: int=1,
    save_name: str = 'results.json',
    best_dpo_model_on_dev: bool = True, 
    idk_prompt: bool = False,
    **kwargs
):
    if prompt_file is not None:
        assert os.path.exists(
            prompt_file
        ), f"Provided Prompt file does not exist {prompt_file}"

        dialogs= read_dialogs_from_file(prompt_file)

    elif not sys.stdin.isatty():
        dialogs = "\n".join(sys.stdin.readlines())
    else:
        print("No user prompt provided. Exiting.")
        sys.exit(1)

    print(f"User dialogs number: {len(dialogs)}")
    print("\n==================================\n")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    if os.path.exists(os.path.join(model_name, 'config.yaml')):
        model = load_dpo_model(model_name, quantization, best_dpo_model_on_dev)
    else:
        model = load_model(model_name, quantization)
    if peft_model:
        model = load_peft_model(model, peft_model)

    if use_fast_kernels:
        """
        Setting 'use_fast_kernels' will enable
        using of Flash Attention or Xformer memory-efficient kernels 
        based on the hardware being used. This would speed up inference when used for batched inputs.
        """
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)   
        except ImportError:
            print("Module 'optimum' not found. Please install 'optimum' it before proceeding.")

    if os.path.exists(os.path.join(model_name, 'config.yaml')):
        # loading DPO model
        with open(os.path.join(model_name, 'config.yaml'), 'r', encoding='utf-8') as f:
            config = yaml.load(f.read(), Loader=yaml.FullLoader)
        tokenizer = LlamaTokenizer.from_pretrained(config['model']['name_or_path'], padding_side='left')
    else:
        tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token
    
    batch_num = math.ceil(len(dialogs) / batch_size)

    generated_results = []
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            chunk_data = dialogs[i*batch_size:(i+1)*batch_size]
            if idk_prompt:
                input_data = ["Answer the following question, and if you don't know the answer, only reply with 'I don't know': {}".format(item['question']) for item in chunk_data]
            else:
                input_data = [item['question'] for item in chunk_data]
            inputs = format_tokens_triviaqa(input_data, tokenizer)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=response_num,
                **kwargs
            )
            
            if response_num > 1:
                for idx in range(len(chunk_data)):
                    chunk_data[idx]['generated_answer'] = []
                    for j in range(response_num):
                        chunk_data[idx]['generated_answer'].append(tokenizer.decode(outputs[idx*response_num+j][inputs.input_ids.shape[1]:], skip_special_tokens=True))
            else:
                for idx in range(len(chunk_data)):
                    chunk_data[idx]['generated_answer'] = tokenizer.decode(outputs[idx][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            generated_results.extend(chunk_data)

    # save results
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)
