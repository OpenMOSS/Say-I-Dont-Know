# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import math
import json

import torch
import torch.nn as nn
import deepspeed
from transformers import LlamaTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from llama_recipes.utils.train_utils import RewardModel

B_INST, E_INST = "[INST]", "[/INST]"

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

def load_state_dict_into_model(model_to_load=None,
                               state_dict=None,
                               start_prefix="",
                               zero_stage=0):

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            if zero_stage == 3:
                # In sharded models, each shard has only part of the full state_dict, so only gather
                # parameters that are in the current state_dict.
                named_parameters = dict(
                    module.named_parameters(prefix=prefix[:-1], recurse=False))
                params_to_gather = [
                    named_parameters[k] for k in state_dict.keys()
                    if k in named_parameters
                ]
                if len(params_to_gather) > 0:
                    # because zero3 puts placeholders in model params, this context
                    # manager gathers (unpartitions) the params of the current layer, then loads from
                    # the state dict and then re-partitions them again
                    with deepspeed.zero.GatheredParameters(params_to_gather,
                                                           modifier_rank=0):
                        if torch.distributed.get_rank() == 0:
                            module._load_from_state_dict(*args)
            else:
                module._load_from_state_dict(*args)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs

def load_auto_model_from_config(config_path):
    model_config = AutoConfig.from_pretrained(config_path) 
    model = AutoModel.from_config(model_config)
    return model

def load_model(tokenizer, ckpt_path, base_model_name_or_path='/cpfs01/shared/public/public_hdd/zhangshuo/ckpt/llama2/llama-2-7b-chat-hf'):
    base_model = load_auto_model_from_config(base_model_name_or_path)
    critic_model = RewardModel(base_model, tokenizer, num_padding_at_beginning=0, compute_fp32_loss=True, score_head_dtype=torch.float32)
    model_ckpt_path = os.path.join(ckpt_path, 'pytorch_model.bin')
    model_ckpt_state_dict = torch.load(model_ckpt_path, map_location='cpu')
    load_state_dict_into_model(critic_model, model_ckpt_state_dict)

    return critic_model.cuda()


def main(
    model_name,
    batch_size: int=1,
    prompt_file: str=None,
    seed: int=42, #seed value for reproducibility
    save_name: str = 'results.json',
    reject_sampling: bool = False,
):
    assert batch_size == 1, "batch_size must be 1"
    with open(prompt_file, 'r', encoding='utf-8') as f:
        dialogs = json.load(f)

    print(f"User dialogs number: {len(dialogs)}")
    print("\n==================================\n")

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

    tokenizer = LlamaTokenizer.from_pretrained(model_name, padding_side='right')
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(tokenizer, model_name)
    
    batch_num = math.ceil(len(dialogs) / batch_size)
    print(len(dialogs))

    refuse_answer = 'This question is beyond the scope of my knowledge, and I am not sure what the answer is.'

    generated_results = []
    with torch.no_grad():
        for i in tqdm(range(batch_num)):
            chunk_data = dialogs[i*batch_size:(i+1)*batch_size]
            if reject_sampling:
                assert len(chunk_data) == 1
                question = chunk_data[0]['question']
                input_question_answers = []
                candidates = list(set(chunk_data[0]['generated_answer']))
                for generated_answer in candidates:
                    input_question_answers.append({'question': question, 'generated_answer': generated_answer})
                inputs = format_tokens_triviaqa_for_ppo_reward(input_question_answers, tokenizer)
                results_scores = model.forward_value(**inputs)['chosen_end_scores']
                max_score_index = torch.argmax(results_scores)
                chunk_data[0]['generated_answer'] = candidates[max_score_index]
            else:
                inputs_normal_answers = format_tokens_triviaqa_for_ppo_reward(chunk_data, tokenizer)
                inputs_refuse_answers = format_tokens_triviaqa_for_ppo_reward(chunk_data, tokenizer, refuse_answer)
                result_normal_answers = model.forward_value(**inputs_normal_answers)
                result_refuse_answers = model.forward_value(**inputs_refuse_answers)
                normal_answers_scores = result_normal_answers['chosen_end_scores']
                refuse_answers_scores = result_refuse_answers['chosen_end_scores']
                
                for idx in range(len(chunk_data)):
                    if normal_answers_scores[idx] > refuse_answers_scores[idx]:
                        chunk_data[idx]['know_or_unknow'] = 'know'
                    else:
                        chunk_data[idx]['know_or_unknow'] = 'unknow'
            
            generated_results.extend(chunk_data)
    # save results
    with open(save_name, 'w', encoding='utf-8') as f:
        json.dump(generated_results, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    fire.Fire(main)
