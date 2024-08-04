import json
from spacy.training import Example
from spacy.scorer import Scorer
from spacy.tokens import Doc
from spacy.vocab import Vocab

V = Vocab()

# Load the tokenized data
with open('hkcancor_segmented.jsonl', 'r', encoding='utf-8') as file:
    data = [json.loads(line) for line in file]

# Load the error data
with open('hkcancor_errors.jsonl', 'r', encoding='utf-8') as error_file:
    error_data = [json.loads(line) for line in error_file]

# Extract references from error data
error_references = {json.dumps(entry['reference'], ensure_ascii=False) for entry in error_data}

# Filter out data with references in error_references
filtered_data = [entry for entry in data if json.dumps(entry['reference'], ensure_ascii=False) not in error_references]

# Use filtered_data for further processing
data = filtered_data


# Prepare the references and predictions
examples = []

for entry in data:
    reference = entry['reference']
    hypothesis = entry['hypothesis']
    predicted = Doc(V, words=hypothesis, spaces=[False for _ in hypothesis])
    target = Doc(V, words=reference, spaces=[False for _ in reference])
    example = Example(predicted, target)
    examples.append(example)

# Calculate the F1 score using Scorer
scorer = Scorer()
results = scorer.score_tokenization(examples)

print(f"Token Accuracy: {results['token_acc']}")
print(f"Token F1 Score: {results['token_f']}")
print(f"Token Precision: {results['token_p']}")
print(f"Token Recall: {results['token_r']}")
