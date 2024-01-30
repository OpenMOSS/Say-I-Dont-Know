import os
import json
import random
import argparse
from tqdm import tqdm
random.seed(42)

KEC_mapping = {
    '0.1': '1.0',
    '0.2': '0.9',
    '0.3': '0.8',
    '0.4': '0.7',
    '0.5': '0.6',
    '0.6': '0.5',
    '0.7': '0.4',
    '0.8': '0.3',
    '0.9': '0.2',
    '1.0': '0.1',
    # 'all refuse': '0.0',
}

prompt = '''Your current knowledge expression confidence level is {}, please answer the user's question:
{}
'''.format

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='/sft_data/llama-2-7b-chat/')
    args = parser.parse_args()
    return args

def process(args):
    root_dir = args.root_dir
    train_data = []
    valid_data = []
    test_data = []

    for i in tqdm(range(1, 11)):
        Ik_threshold = str(i / 10)
        train_file_name = os.path.join(root_dir, 'triviaqa_train_threshold_{}_sft_data.json'.format(Ik_threshold))
        valid_file_name = os.path.join(root_dir, 'triviaqa_valid_threshold_{}_sft_data.json'.format(Ik_threshold))
        test_file_name = os.path.join(root_dir, 'triviaqa_test_threshold_{}_sft_data.json'.format(Ik_threshold))
        with open(train_file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                item['confidence_level'] = KEC_mapping[Ik_threshold]
                item['question'] = prompt(item['confidence_level'], item['question'])
                train_data.append(item)

        with open(valid_file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                item['confidence_level'] = KEC_mapping[Ik_threshold]
                item['question'] = prompt(item['confidence_level'], item['question'])
                valid_data.append(item)

        with open(test_file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                item['confidence_level'] = KEC_mapping[Ik_threshold]
                item['question'] = prompt(item['confidence_level'], item['question'])
                test_data.append(item)

    with open(os.path.join(root_dir, 'triviaqa_dev_tp1.0_10responses_with_em_labels.json'), 'r', encoding='utf-8') as f:
        data = json.load(f)
    for item in data:
        new_sample_0 = {
            "question_id": item['question_id'],
            "confidence_level": "0.0",
            "question": prompt("0.0", item['question']),
            'answer': "This question is beyond the scope of my knowledge, and I am not sure what the answer is.",
        }
        test_data.append(new_sample_0)

    with open(os.path.join(root_dir, 'triviaqa_train_threshold_0.1_sft_data.json'), 'r', encoding='utf-8') as f:
        truely_train_data = json.load(f)
    for item in truely_train_data:
        new_sample_0 = {
            "question_id": item['question_id'],
            "confidence_level": "0.0",
            "question": prompt("0.0", item['question']),
            'answer': "This question is beyond the scope of my knowledge, and I am not sure what the answer is.",
        }
        train_data.append(new_sample_0)

    with open(os.path.join(root_dir, 'triviaqa_valid_threshold_0.1_sft_data.json'), 'r', encoding='utf-8') as f:
        truely_valid_data = json.load(f)
    for item in truely_valid_data:
        new_sample_0 = {
            "question_id": item['question_id'],
            "confidence_level": "0.0",
            "question": prompt("0.0", item['question']),
            'answer': "This question is beyond the scope of my knowledge, and I am not sure what the answer is.",
        }
        valid_data.append(new_sample_0)

    random.shuffle(train_data)
    random.shuffle(valid_data)
    # save data
    with open(os.path.join(root_dir, 'triviaqa_train_hir_data.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(root_dir, 'triviaqa_valid_hir_data.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, indent=2, ensure_ascii=False)

    with open(os.path.join(root_dir, 'triviaqa_test_hir_data.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    args = get_args()
    process(args)