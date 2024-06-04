import torch
import json
from sklearn.neighbors import NearestNeighbors
from utils import normalize

# Function to compute vector for a word
def compute_vec(word, combined_vecs):
    vecs = [combined_vecs[c] if c in combined_vecs else torch.zeros_like(next(iter(combined_vecs.values()))) for c in word]
    return torch.mean(torch.stack(vecs), dim=0)

# Function to find nearest neighbors
def find_k_nearest_neighbors(query, combined_vecs, variant_vecs, k=10):
    # Compute vector for the query
    query_vec = compute_vec(query, combined_vecs).unsqueeze(0).view(1, -1).numpy()  # Ensure query_vec is 2D
    
    # Get all vectors and labels
    all_vecs = torch.cat([v.view(1, -1) for v in variant_vecs.values()]).numpy()  # Ensure all_vecs is 2D
    labels = list(variant_vecs.keys())
    
    # Fit the nearest neighbors model
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(all_vecs)
    distances, indices = nbrs.kneighbors(query_vec)
    
    # Get the nearest neighbors and their distances
    neighbors = [(labels[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return neighbors

if __name__ == '__main__':
    # Load data
    with open('../dict.json', 'r') as f:
        data = json.load(f)

    # Extract variants
    variants = set()
    for entry in data.values():
        for variant in entry.get('variants', []):
            variants.add(variant.get('w', ''))

    # Load combined vectors
    combined_vecs = torch.load('combined_vecs.pt')

    # Compute vectors for all variants
    variant_vecs = {variant: compute_vec(variant) for variant in variants}

    # Example usage
    query = '景廣'
    k = 10
    nearest_neighbors = find_k_nearest_neighbors(query, combined_vecs, variant_vecs, k)
    for neighbor, distance in nearest_neighbors:
        print(f"Word: {neighbor}, Distance: {distance}")
