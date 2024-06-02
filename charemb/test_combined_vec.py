import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def load_vectors(vec_type):
    if vec_type == 'bert':
        return torch.load('bert_vecs.pt')
    elif vec_type == 'char':
        return torch.load('char_vecs.pt')
    elif vec_type == 'whisper':
        return torch.load('whisper_vecs.pt')
    elif vec_type == 'combined':
        return torch.load('combined_vecs.pt')
    else:
        raise ValueError("Invalid vector type. Choose from 'bert', 'char', 'whisper', 'combined'.")

def find_k_nearest_neighbors(character, vec_type, k=5):
    vectors = load_vectors(vec_type)
    
    if character not in vectors:
        raise ValueError(f"Character '{character}' not found in {vec_type} vectors.")
    
    char_vec = vectors[character].unsqueeze(0).view(1, -1).numpy()  # Ensure char_vec is 2D
    all_vecs = torch.cat([v.view(1, -1) for v in vectors.values()]).numpy()  # Ensure all_vecs is 2D
    labels = list(vectors.keys())
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_vecs)
    distances, indices = nbrs.kneighbors(char_vec)
    
    neighbors = [(labels[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return neighbors

if __name__ == '__main__':
    character = '轟'
    jyutping = 'gwang1'
    k = 5
    vec_types = ['bert', 'char', 'whisper', 'combined']
    
    for vec_type in vec_types:
        if vec_type == 'whisper':
            neighbors = find_k_nearest_neighbors(jyutping, vec_type, k)
            print(f"The {k} nearest neighbors of '{jyutping}' using {vec_type} vectors are:")
        else:
            neighbors = find_k_nearest_neighbors(character, vec_type, k)
            print(f"The {k} nearest neighbors of '{character}' using {vec_type} vectors are:")
        
        for neighbor, distance in neighbors:
            print(f"Character: {neighbor}, Distance: {distance}")
        print("\n")
