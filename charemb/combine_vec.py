import torch
import umap
from cluster import plot_embeddings
from utils import normalize
import sys

combine_method = sys.argv[1] if len(sys.argv) > 1 else 'add'

print(f'Using combine method {combine_method}')

# Load the precomputed vectors
bert_vecs = torch.load('bert_vecs.pt')
char_vecs = torch.load('char_vecs.pt')
whisper_vecs = torch.load('whisper_char_vecs.pt')

assert(sorted(list(bert_vecs.keys())) == sorted(list(char_vecs.keys())))

bert_vecs = {key: bert_vecs[key] for key in sorted(bert_vecs.keys())}
char_vecs = {key: char_vecs[key] for key in sorted(char_vecs.keys())}


def reduce_dimension(vec_dict, target_n_components=100):
    vecs = torch.stack(list(vec_dict.values())).cpu().squeeze().numpy()
    vecs = umap.UMAP(n_components=target_n_components).fit_transform(vecs)
    return {key: torch.tensor(vec).unsqueeze(0) for key, vec in zip(bert_vecs.keys(), vecs)}

bert_vecs = reduce_dimension(bert_vecs)
char_vecs = reduce_dimension(char_vecs)
whisper_vecs = reduce_dimension(whisper_vecs)

combined_vecs = {}

for char, bert_vec in bert_vecs.items():
    bert_vec = normalize(bert_vec)
    char_vec = normalize(char_vecs[char])
    whisper_vec = normalize(whisper_vecs[char])
    match combine_method:
        case 'add':
            combined_vecs[char] = bert_vec + char_vec + whisper_vec
        case 'concat' | 'concat_umap':
            combined_vecs[char] = torch.cat((whisper_vec, bert_vec, char_vec), dim=1)

match combine_method:
    case 'concat_umap':
        combined_vecs = reduce_dimension(combined_vecs)

torch.save(combined_vecs, f'combined_vecs_{combine_method}.pt')
embeddings = torch.cat(list(combined_vecs.values())).cpu().numpy()
labels = list(combined_vecs.keys())
plot_embeddings(embeddings, labels)
