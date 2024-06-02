from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as transforms
import numpy as np

# https://stackoverflow.com/a/77749307/6798201
def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

# Custom Dataset class
class ChineseCharacterDataset(Dataset):
    def __init__(self, characters, char_labels, jyutping_labels, font_path, transform=None):
        self.characters = characters
        self.char_labels = char_labels
        self.jyutping_labels = jyutping_labels
        self.font = ImageFont.truetype(font_path, size=64)
        self.transform = transform

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        character = self.characters[idx]
        image = self.render_character(character)
        if self.transform:
            image = self.transform(image)
        char_label = self.char_labels[idx]
        jyutping_label = self.jyutping_labels[idx]  # Should be a binary vector
        return image, character, char_label, jyutping_label

    def render_character(self, character):
        # Create a blank image with white background
        image = Image.new('L', (64, 64), color='white')
        draw = ImageDraw.Draw(image)
        # Calculate width and height of the character to center it
        width, height = textsize(text=character, font=self.font)
        # Position the character at the center of the image
        draw.text(((64 - width) / 2, (64 - height) / 2), character, fill='black', font=self.font)
        return image


import json

def get_dataset():
    with open('../data/char_jyutpings/charlist_processed.json', 'r') as f:
        char_jyutpings = json.load(f)

    characters = list(char_jyutpings.keys())

    char_labels = np.array(range(len(characters)))
    jyutping_labels = []
    
    all_jyutpings = set()
    for char, jyutpings in char_jyutpings.items():
        all_jyutpings.update(jyutpings.keys())
    all_jyutpings = sorted(list(all_jyutpings))
    
    for char, jyutpings in char_jyutpings.items():
        jyutping_vector = [0] * len(all_jyutpings)
        for jyutping, _ in jyutpings.items():
            jyutping_vector[all_jyutpings.index(jyutping)] = 1
        jyutping_labels.append(jyutping_vector)
    jyutping_labels = np.array(jyutping_labels)

    # Font path
    font_path = 'ChironHeiHK-R.ttf'

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset and DataLoader
    dataset = ChineseCharacterDataset(characters, char_labels, jyutping_labels, font_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_char_classes = len(np.unique(char_labels))  # Adjust according to your dataset
    num_jyutping_classes = jyutping_labels.shape[1]  # Number of Jyutping classes

    return (dataloader, num_char_classes, num_jyutping_classes)
