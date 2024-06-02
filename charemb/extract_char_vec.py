import torch
from model import MultiTaskCNN
from dataset import get_dataset

if __name__ == '__main__':
    # Load the dataset
    dataloader, num_char_classes = get_dataset()

    # Load the model
    model = MultiTaskCNN(num_char_classes)
    model.load_state_dict(torch.load('charemb.pth'))
    model.eval()

    # Extract embeddings
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, characters, char_labels, _ in dataloader:
            features = model.features(inputs)
            features = features.view(features.size(0), -1)
            embs = model.fc(features)
            embeddings.extend(embs)
            labels.extend(characters)

    embeddings_dict = {label: embedding.cpu() for label, embedding in zip(labels, embeddings)}
    torch.save(embeddings_dict, 'char_vecs.pt')
