from search_demo import find_k_nearest_neighbors
import torch
from tqdm import tqdm
import sys
import random
import json

random.seed(42)

vec_type = sys.argv[1] if len(sys.argv) > 1 else 'add'

print(f'Using vector type {vec_type}')

with open('eval_pairs.json', 'r') as f:
    eval_data = json.load(f)

print(f'Evaluating with {len(eval_data)} unique character pairs')

match vec_type:
    case 'bert':
        vec_name = 'bert_vecs'
    case 'char':
        vec_name = 'char_vecs'
    case 'whisper':
        vec_name = 'whisper_char_vecs'
    case _:
        vec_name = f'combined_vecs_{vec_type}'

vecs = torch.load(f'{vec_name}.pt')

found_pairs = []

for (char1, char2) in tqdm(eval_data):
    if random.random() < 0.5:
        char1, char2 = char2, char1
    results = find_k_nearest_neighbors(char1, vecs, vecs, k=11)
    for variant, distance in results:
        if variant == char2:
            found_pairs.append((char1, variant))

print(f'Score: {len(found_pairs)}')

with open(f'found_pairs_{vec_type}.json', 'w') as f:
    json.dump(found_pairs, f, ensure_ascii=False)
