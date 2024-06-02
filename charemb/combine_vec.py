import torch
from dataset import get_dataset
from tqdm import tqdm
from cluster import plot_embeddings
import json

# Check for MPS device
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f'Using device: {device}')

# Load the precomputed vectors
bert_vecs = torch.load('bert_vecs.pt', map_location=device)
char_vecs = torch.load('char_vecs.pt', map_location=device)
whisper_vecs = torch.load('whisper_vecs.pt', map_location=device)

# Load the dataset
dataloader, num_char_classes = get_dataset()

combined_vecs = {}

def normalize(vec):
    norm = torch.norm(vec)
    return vec / norm if norm != 0 else vec

import pandas as pd
import torch.nn as nn
import torch.optim as optim

# Load mistakes.csv
mistakes_df = pd.read_csv('mistakes.csv')

# Prepare training data
train_data = []
for _, row in mistakes_df.iterrows():
    mistake_word, correct_words = row['mistake'], row['correct'].split(' ')
    for correct_word in correct_words:
        if len(mistake_word) == len(correct_word):
            for m_char, c_char in zip(mistake_word, correct_word):
                if m_char != c_char:
                    train_data.append((m_char, c_char))
                    train_data.append((c_char, m_char))

# Define the attention model
class AttentionModel(nn.Module):
    def __init__(self, input_dim, num_heads=1):
        super(AttentionModel, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads)

    def forward(self, bert_vec, char_vec, whisper_vec):
        # Combine the input vectors along the sequence dimension (seq_len x batch_size x embed_dim)
        combined = torch.stack([bert_vec, char_vec, whisper_vec], dim=0)  # Shape: (3, batch_size, input_dim)
        # MultiheadAttention expects the input in the format (seq_len, batch_size, embed_dim)
        attn_output, attn_weights = self.multihead_attn(combined, combined, combined)
        # We only need the output, not the attention weights
        weighted_sum = attn_output.sum(dim=0)  # Summing over the sequence length dimension
        return weighted_sum

# Initialize the model, loss function, and optimizer
input_dim = next(iter(bert_vecs.values())).shape[1]
model = AttentionModel(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Prepare vectors for training
def get_vector(char, vec_dict):
    if char in vec_dict:
        return vec_dict[char].to(device)
    else:
        return torch.zeros_like(next(iter(vec_dict.values()))).to(device)

with open('../data/char_jyutpings/charlist_processed.json', 'r') as f:
    char_jyutpings = json.load(f)

def get_jyutpings(char):
    return char_jyutpings.get(char, [])

def get_vecs(character, bert_vecs, char_vecs, whisper_vecs):
    bert_vec = get_vector(character, bert_vecs)
    char_vec = get_vector(character, char_vecs)
    whisper_vec = torch.zeros_like(bert_vec)
    jyutpings = get_jyutpings(character)
    for jyutping in jyutpings:
        if jyutping in whisper_vecs:
            whisper_vec += normalize(whisper_vecs[jyutping].to(device))
    whisper_vec = normalize(whisper_vec)
    return bert_vec, char_vec, whisper_vec


# Training loop
for epoch in range(3):  # Number of epochs can be adjusted
    for m_char, c_char in train_data:
        optimizer.zero_grad()
        with torch.no_grad():
            target_vec = model(*get_vecs(m_char, bert_vecs, char_vecs, whisper_vecs))
        output_vec = model(*get_vecs(c_char, bert_vecs, char_vecs, whisper_vecs))
        loss = criterion(output_vec, target_vec)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} loss: {loss.item()}')

with torch.no_grad():
    # Use the trained model to combine vectors
    for inputs, characters, char_labels, jyutpings in tqdm(dataloader):
        for i, char in enumerate(characters):
            bert_vec = get_vector(char, bert_vecs)
            char_vec = get_vector(char, char_vecs)
            whisper_vec = torch.zeros_like(bert_vec)
            for jyutping in jyutpings[i].split(','):
                if jyutping in whisper_vecs:
                    whisper_vec += normalize(whisper_vecs[jyutping].to(device))
            
            whisper_vec = normalize(whisper_vec)
            combined_vec = model(bert_vec, char_vec, whisper_vec)
            combined_vecs[char] = combined_vec.cpu()

# Save the combined vectors
torch.save(combined_vecs, 'combined_vecs.pt')
embeddings = torch.cat(list(combined_vecs.values())).cpu().numpy()
labels = list(combined_vecs.keys())
plot_embeddings(embeddings, labels)
