import torch
from dataset import get_dataset
from tqdm import tqdm
from cluster import plot_embeddings

# Load the precomputed vectors
bert_vecs = torch.load('bert_vecs.pt')
char_vecs = torch.load('char_vecs.pt')
whisper_vecs = torch.load('whisper_vecs.pt')

# Load the dataset
dataloader, num_char_classes = get_dataset()

combined_vecs = {}

for inputs, characters, char_labels, jyutpings in tqdm(dataloader):
    for i, char in enumerate(characters):
        # Get BERT and character vectors
        if char in bert_vecs:
            bert_vec = bert_vecs[char]
        else:
            print(f"Warning: BERT vector for character '{char}' not found.")
            bert_vec = torch.zeros_like(next(iter(bert_vecs.values())))

        if char in char_vecs:
            char_vec = char_vecs[char]
        else:
            print(f"Warning: Character vector for character '{char}' not found.")
            char_vec = torch.zeros_like(next(iter(char_vecs.values())))
        
        # Concatenate whisper vectors for all jyutpings of the character
        jyutping_list = jyutpings[i].split(',')
        whisper_vec = torch.zeros_like(bert_vec)
        for jyutping in jyutping_list:
            if jyutping in whisper_vecs:
                whisper_vec += whisper_vecs[jyutping]
            else:
                print(f"Warning: Whisper vector for jyutping '{jyutping}' not found.")
        # Combine the vectors by adding them
        combined_vec = bert_vec + char_vec + whisper_vec
        combined_vecs[char] = combined_vec

# Save the combined vectors
torch.save(combined_vecs, 'combined_vecs.pt')

embeddings = torch.cat(list(combined_vecs.values())).cpu().numpy()
labels = list(combined_vecs.keys())
plot_embeddings(embeddings, labels)
