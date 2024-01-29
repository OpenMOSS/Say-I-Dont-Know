import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_name', type=str, default='preference_data/llama-2-7b-chat/triviaqa_train_and_valid_llama2_7b_chat_threshold_1.0_preference_pairs.json')
    parser.add_argument('--save_name', type=str, default='preference_data/llama-2-7b-chat/triviaqa_train_and_valid_llama2_7b_chat_threshold_1.0_preference_pairs_for_ppo_reward.json')
    args = parser.parse_args()
    return args

def process_data(args):
    with open(args.file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preference_pairs = {
        'train': [],
        'test': []
    }
    for item in data['train']:
        for single_pairs in item['preference_pairs']:
            new_sample = {
                'question': item['question'],
                'positive_answer': single_pairs['positive_answer'],
                'negative_answer': single_pairs['negative_answer'],
            }
            preference_pairs['train'].append(new_sample)
    for item in data['test']:
        for single_pairs in item['preference_pairs']:
            new_sample = {
                'question': item['question'],
                'positive_answer': single_pairs['positive_answer'],
                'negative_answer': single_pairs['negative_answer'],
            }
            preference_pairs['test'].append(new_sample)

    print('train preference pairs num:', len(preference_pairs['train']))
    print('valid preference pairs num:', len(preference_pairs['test']))
    with open(args.save_name, 'w', encoding='utf-8') as f:
        json.dump(preference_pairs, f, indent=2, ensure_ascii=False)

if __name__ == '__main__':
    args = get_args()
    process_data(args)