import os
import json
import random
import argparse
from tqdm import tqdm

random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama-2-7b-chat')
    parser.add_argument('--threshold', type=float, default=1.0)
    args = parser.parse_args()
    return args

def process(args):
    refuse_answer = 'This question is beyond the scope of my knowledge, and I am not sure what the answer is.'

    def construct_pos_neg(original_data):
        positive_sample = []
        negative_sample = []

        for item in tqdm(original_data):
            new_sample = {
                'question_id': item['question_id'],
                'question': item['question'],
            }
            correct_count = 0
            correct_answers = []
            for candidate in item['generated_answer']:
                if candidate['True_or_False'] == 'True':
                    correct_count += 1
                    correct_answers.append(candidate['generated_answer'])
            acc = correct_count / len(item['generated_answer'])
            if acc >= args.threshold:
                new_sample['answer'] = random.choice(correct_answers)
                positive_sample.append(new_sample)
            else:
                new_sample['answer'] = refuse_answer
                negative_sample.append(new_sample)

        return positive_sample, negative_sample

    # process train and valid data
    train_file_path = os.path.join('sft_data', args.model_name, 'triviaqa_train_tp1.0_10responses_with_em_labels.json')
    with open(train_file_path, 'r') as f:
        original_data = json.load(f)
    
    validation_ratio = 0.1
    train_positive_sample, train_negative_sample = construct_pos_neg(original_data)
    pos_train_num = int(len(train_positive_sample) * (1 - 0.1))
    neg_train_num = int(len(train_negative_sample) * (1 - 0.1))
    random.shuffle(train_positive_sample)
    random.shuffle(train_negative_sample)
    train_set = train_positive_sample[:pos_train_num] + train_negative_sample[:neg_train_num]
    valid_set = train_positive_sample[pos_train_num:] + train_negative_sample[neg_train_num:]
    random.shuffle(train_set)
    random.shuffle(valid_set)
    print('Number of train samples:', len(train_set))
    print('Number of valid samples:', len(valid_set))

    with open(os.path.join('sft_data', args.model_name, 'triviaqa_train_threshold_10_sft_data.json'), 'w') as f:
        json.dump(train_set, f, indent=2, ensure_ascii=False)
    with open(os.path.join('sft_data', args.model_name, 'triviaqa_valid_threshold_10_sft_data.json'), 'w') as f:
        json.dump(valid_set, f, indent=2, ensure_ascii=False)

    # process test data
    test_file_path = os.path.join('sft_data', args.model_name, 'triviaqa_dev_tp1.0_10responses_with_em_labels.json')
    with open(test_file_path, 'r') as f:
        original_data = json.load(f)
    test_positive_sample, test_negative_sample = construct_pos_neg(original_data)
    test_set = test_positive_sample + test_negative_sample
    random.shuffle(test_set)
    print('Number of test samples:', len(test_set))
    
    with open(os.path.join('sft_data', args.model_name, 'triviaqa_test_threshold_10_sft_data.json'), 'w') as f:
        json.dump(test_set, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    args = get_args()
    process(args)