import torch

vec_map = torch.load("char_vecs.pt")

cos = torch.nn.CosineSimilarity(dim=1)

# For each character, find another char with the highest cosine similarity, ensure we don't compare the same char
for char, vec in vec_map.items():
    similarities = [(cos(vec, vec2).item(), char2) for char2, vec2 in vec_map.items() if char != char2]
    # Print the best 10 similarities
    print(char + "," + "".join([x[1] for x in sorted(similarities, reverse=True)[:20]]))
