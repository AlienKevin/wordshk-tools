import torch
from model import MultiTaskCNN
from dataset import get_dataset

if __name__ == '__main__':
    # Load the dataset
    dataloader, num_char_classes = get_dataset(expanded_charset=True, augmented=False)

    # Load the model
    model = MultiTaskCNN(num_char_classes, fc_width=20)
    model.load_state_dict(torch.load('charemb.pth'))
    model.eval()

    # Extract embeddings
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, characters, char_labels, _, _ in dataloader:
            features = model.features(inputs)
            features = features.view(features.size(0), -1)
            embs = model.fc1(features)
            embs = model.fc2(embs)
            embs = torch.nn.functional.normalize(embs)
            embeddings.extend(embs)
            labels.extend(characters)

    embeddings_dict = {label: embedding.cpu().unsqueeze(0) for label, embedding in zip(labels, embeddings)}
    torch.save(embeddings_dict, 'char_vecs.pt')
