import pandas as pd
import json

# Load mistakes.csv
mistakes_df = pd.read_csv('mistakes.csv')

# Prepare training data
eval_data = set()
for _, row in mistakes_df.iterrows():
    mistake_word, correct_words = row['mistake'], row['correct'].split(' ')
    for correct_word in correct_words:
        if len(mistake_word) == len(correct_word):
            for m_char, c_char in zip(mistake_word, correct_word):
                if m_char != c_char:
                    eval_data.add((m_char, c_char))

def filter(vec_type):
    with open(f'found_pairs_{vec_type}.json', 'r') as f:
        found_pairs = [(char1, char2) for char1, char2 in json.load(f)]
        filtered_pairs = [(char1, char2) for char1, char2 in found_pairs if (char1, char2) in eval_data or (char2, char1) in eval_data]
        print(f"Number of characters before filtering: {len(found_pairs)}")
        print(f"Number of characters after filtering: {len(filtered_pairs)}/{len(eval_data)}")
        unrecognized_pairs = [(char1, char2) for (char1, char2) in sorted(list(eval_data)) if (char1, char2) not in found_pairs and (char2, char1) not in found_pairs]
        print(len(unrecognized_pairs))
        print(unrecognized_pairs)

for vec_type in ['bert', 'char', 'whisper', 'add', 'concat', 'concat_umap']:
    print(vec_type)
    filter(vec_type)
