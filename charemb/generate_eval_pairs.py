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

with open('variants.txt', 'r') as f:
    for line in f.readlines():
        variants = line.split(' ')
        for variant1 in variants:
            for variant2 in variants:
                if variant1 != variant2:
                    for char1, char2 in zip(variant1, variant2):
                        if char1 != char2:
                            eval_data.add((char1, char2))

with open('eval_pairs.json', 'w') as f:
    json.dump(sorted(list(eval_data)), f, ensure_ascii=False)
