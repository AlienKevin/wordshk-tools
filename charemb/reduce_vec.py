import torch
import umap

char_vecs = torch.load('char_vecs.pt')
char_vecs = {key: char_vecs[key] for key in sorted(char_vecs.keys())}

def reduce_dimension(vec_dict, target_n_components=10):
    vecs = torch.stack(list(vec_dict.values())).cpu().squeeze().numpy()
    vecs = umap.UMAP(n_components=target_n_components).fit_transform(vecs)
    return {key: torch.nn.functional.normalize(torch.tensor(vec).unsqueeze(0)) for key, vec in zip(char_vecs.keys(), vecs)}

char_vecs = reduce_dimension(char_vecs)
torch.save(char_vecs, 'char_vecs_10.pt')
