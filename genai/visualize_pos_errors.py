import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['deepseek-chat', 'deepseek-coder', 'gpt-4o', 'gpt-4o-mini', 'qwen-max', 'qwen-plus', 'qwen-turbo'], required=True, help='Model to use for POS tagging')
parser.add_argument('--prompt_version', type=str, choices=['v1', 'v2'], required=True, help='Prompt version to use for POS tagging')
parser.add_argument('--eval_dataset', type=str, choices=['hkcancor', 'ud_yue'], required=True, help='Dataset to evaluate POS tagging on')
parser.add_argument('--segmentation_given', type=bool, default=False, help='Whether the segmentation is given')
parser.add_argument('--maximize_diversity', type=bool, default=False, help='Whether to maximize in-context example diversity')
args = parser.parse_args()

# Read the error data
with open(f'outputs/pos_errors_{args.eval_dataset}_{args.model}_prompt_{args.prompt_version}{"_max_diversity" if args.maximize_diversity else ""}{"_segmentation_given" if args.segmentation_given else ""}.jsonl', 'r', encoding='utf-8') as error_file:
    error_data = [json.loads(line) for line in error_file]

with open(f'outputs/pos_{args.eval_dataset}_{args.model}_prompt_{args.prompt_version}{"_max_diversity" if args.maximize_diversity else ""}{"_segmentation_given" if args.segmentation_given else ""}.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

import difflib

red = lambda text: f"\033[38;2;255;0;0m{text}\033[38;2;255;255;255m"
green = lambda text: f"\033[38;2;0;255;0m{text}\033[38;2;255;255;255m"
blue = lambda text: f"\033[38;2;0;0;255m{text}\033[38;2;255;255;255m"
white = lambda text: f"\033[38;2;255;255;255m{text}\033[38;2;255;255;255m"

def get_edits_string(old, new):
    result = ""
    codes = difflib.SequenceMatcher(a=old, b=new).get_opcodes()
    for code in codes:
        if code[0] == "equal": 
            result += white(old[code[1]:code[2]])
        elif code[0] == "delete":
            result += red(old[code[1]:code[2]])
        elif code[0] == "insert":
            result += green(new[code[3]:code[4]])
        elif code[0] == "replace":
            result += (red(old[code[1]:code[2]]) + green(new[code[3]:code[4]]))
    return result

# Visualize the error between the reference and hypothesis
for entry in error_data:
    # Filter out entries with the specific error
    if entry['error'] != "Segmentation result does not match the input sentence":
        print('ERROR:', entry['error'])
        print(entry)
        continue

    reference_tokens = [token for token, pos in entry['reference']]
    hypothesis_tokens = [token for token, pos in json.loads(entry['hypothesis'])['pos_tagged_words']]
    
    reference_sentence = "".join(reference_tokens)
    hypothesis_sentence = "".join(hypothesis_tokens)
    
    diff = get_edits_string(hypothesis_sentence, reference_sentence)
    
    print(diff)

for entry in data:
    if any(e['reference'] == entry['reference'] for e in error_data) or entry['reference'] == entry['hypothesis']:
        continue
    else:
        reference_tokens = [f"{token}:{pos}" for token, pos in entry['reference']]
        hypothesis_tokens = [f"{token}:{pos}" for token, pos in entry['hypothesis']]
        
        reference_sentence = " ".join(reference_tokens)
        hypothesis_sentence = " ".join(hypothesis_tokens)
        
        diff = get_edits_string(hypothesis_sentence, reference_sentence)
        
        print(diff)
