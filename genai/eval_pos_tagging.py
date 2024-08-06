import json
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.vocab import Vocab
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, choices=['deepseek-chat', 'gpt-4o', 'gpt-4o-mini'], required=True, help='Model to use for POS tagging')
parser.add_argument('--prompt_version', type=str, choices=['v1', 'v2'], required=True, help='Prompt version to use for POS tagging')
parser.add_argument('--eval_dataset', type=str, choices=['hkcancor', 'ud_yue'], required=True, help='Dataset to evaluate POS tagging on')
args = parser.parse_args()

V = Vocab()

# Load the tokenized data
with open(f'outputs/pos_{args.eval_dataset}_{args.model}_prompt_{args.prompt_version}.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# Load the error data
with open(f'outputs/pos_errors_{args.eval_dataset}_{args.model}_prompt_{args.prompt_version}.jsonl', 'r', encoding='utf-8') as error_file:
    error_data = [json.loads(line) for line in error_file]

# Extract references from error data
error_references = {json.dumps(entry['reference'], ensure_ascii=False) for entry in error_data}

# Filter out data with references in error_references
filtered_data = [entry for entry in data if json.dumps(entry['reference'], ensure_ascii=False) not in error_references]

# Use filtered_data for further processing
data = filtered_data


# Prepare the references and predictions
examples = []

# Patches https://github.com/jacksonllee/pycantonese/issues/48
def patch_pycantonese_tag_bug(tag):
    if tag == "V":
        return "VERB"
    else:
        return tag

for entry in data:
    reference = entry['reference']
    hypothesis = entry['hypothesis']
    predicted = Doc(V, words=[x[0] for x in hypothesis], spaces=[False for _ in hypothesis], pos=[x[1] for x in hypothesis])
    target = Doc(V, words=[x[0] for x in reference], spaces=[False for _ in reference], pos=[patch_pycantonese_tag_bug(x[1]) for x in reference])
    example = Example(predicted, target)
    examples.append(example)

# Calculate the F1 score using Scorer
scorer = Scorer()
results = scorer.score(examples)

print(f"POS Tagging Accuracy: {results['pos_acc']}")
print(f"Token Accuracy: {results['token_acc']}")
print(f"Token F1 Score: {results['token_f']}")
print(f"Token Precision: {results['token_p']}")
print(f"Token Recall: {results['token_r']}")
