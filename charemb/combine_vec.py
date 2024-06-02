import torch
from dataset import get_dataset
from tqdm import tqdm
from cluster import plot_embeddings
import json

# Load the precomputed vectors
bert_vecs = torch.load('bert_vecs.pt')
char_vecs = torch.load('char_vecs.pt')
whisper_vecs = torch.load('whisper_vecs.pt')

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

# Define the attention model
class AttentionModel(nn.Module):
    def __init__(self, input_dim):
        super(AttentionModel, self).__init__()
        self.attention = nn.Linear(input_dim, 1)
    
    def forward(self, bert_vec, char_vec, whisper_vec):
        # print(f'BERT vector dimensions: {bert_vec.shape}')
        # print(f'Character vector dimensions: {char_vec.shape}')
        # print(f'Whisper vector dimensions: {whisper_vec.shape}')
        combined = torch.stack([bert_vec, char_vec, whisper_vec], dim=0)
        # print(f'Combined vector dimensions: {combined.shape}')
        attn_weights = torch.softmax(self.attention(combined), dim=0)
        weighted_sum = torch.sum(attn_weights * combined, dim=0)
        return weighted_sum

# Initialize the model, loss function, and optimizer
input_dim = next(iter(bert_vecs.values())).shape[1]
model = AttentionModel(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Prepare vectors for training
def get_vector(char, vec_dict):
    if char in vec_dict:
        return vec_dict[char]
    else:
        return torch.zeros_like(next(iter(vec_dict.values())))


with open('../data/char_jyutpings/charlist_processed.json', 'r') as f:
    char_jyutpings = json.load(f)


def get_jyutpings(char):
    return char_jyutpings.get(char, [])

# Training loop
for epoch in range(5):  # Number of epochs can be adjusted
    for m_char, c_char in train_data:
        bert_vec = get_vector(m_char, bert_vecs)
        char_vec = get_vector(m_char, char_vecs)
        whisper_vec = torch.zeros_like(bert_vec)
        jyutpings = get_jyutpings(m_char)
        for jyutping in jyutpings:
            if jyutping in whisper_vecs:
                whisper_vec += normalize(whisper_vecs[jyutping])
        
        whisper_vec = normalize(whisper_vec)
        target_vec = get_vector(c_char, char_vecs)
        
        optimizer.zero_grad()
        output_vec = model(bert_vec, char_vec, whisper_vec)
        loss = criterion(output_vec, target_vec)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch} loss: {loss}')

with torch.no_grad():
    # Use the trained model to combine vectors
    for inputs, characters, char_labels, jyutpings in tqdm(dataloader):
        for i, char in enumerate(characters):
            bert_vec = get_vector(char, bert_vecs)
            char_vec = get_vector(char, char_vecs)
            whisper_vec = torch.zeros_like(bert_vec)
            for jyutping in jyutpings[i].split(','):
                if jyutping in whisper_vecs:
                    whisper_vec += normalize(whisper_vecs[jyutping])
            
            whisper_vec = normalize(whisper_vec)
            combined_vec = model(bert_vec, char_vec, whisper_vec)
            combined_vecs[char] = combined_vec

# Save the combined vectors
torch.save(combined_vecs, 'combined_vecs.pt')
embeddings = torch.cat(list(combined_vecs.values())).cpu().numpy()
labels = list(combined_vecs.keys())
plot_embeddings(embeddings, labels)
