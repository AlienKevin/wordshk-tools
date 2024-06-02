import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from model import MultiTaskCNN
from dataset import get_dataset
from sklearn.manifold import TSNE
from PIL import ImageFont
from matplotlib import font_manager as fm

# Load the dataset
dataloader, num_char_classes, num_jyutping_classes = get_dataset()

# Load the model
model = MultiTaskCNN(num_char_classes, num_jyutping_classes)
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
        embeddings.extend(embs.unsqueeze(0))
        labels.extend(characters)

embeddings = torch.cat(embeddings).cpu().numpy()

# Perform clustering
kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
cluster_labels = kmeans.labels_

# Reduce dimensions for visualization using t-SNE
tsne = TSNE(n_components=2, random_state=0)
reduced_embeddings = tsne.fit_transform(embeddings)

# Load the font
font_path = 'ChironHeiHK-R.ttf'
font = ImageFont.truetype(font_path, size=12)

custom_font = fm.FontProperties(fname=font_path)

# Add the font to Matplotlib's font manager
fm.fontManager.addfont(font_path)

# Set the font globally
plt.rcParams['font.family'] = custom_font.get_name()
plt.rcParams['font.sans-serif'] = custom_font.get_name()

# Plot the clusters with characters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis')
plt.colorbar(scatter)
plt.title('Character Embeddings Clustering using t-SNE')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

# Annotate the plot with characters
for i, label in enumerate(labels):
    plt.text(reduced_embeddings[i, 0], reduced_embeddings[i, 1], label, fontproperties=custom_font, fontsize=12)
plt.show()
