from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import torchvision.transforms as transforms
import numpy as np
import json

# https://stackoverflow.com/a/77749307/6798201
def textsize(text, font):
    im = Image.new(mode="P", size=(0, 0))
    draw = ImageDraw.Draw(im)
    _, _, width, height = draw.textbbox((0, 0), text=text, font=font)
    return width, height

# Custom Dataset class
class ChineseCharacterDataset(Dataset):
    def __init__(self, characters, char_labels, jyutpings, font_path, transform=None):
        self.characters = characters
        self.char_labels = char_labels
        self.jyutpings = jyutpings
        self.font = ImageFont.truetype(font_path, size=60)
        self.transform = transform

    def __len__(self):
        return len(self.characters)

    def __getitem__(self, idx):
        character = self.characters[idx]
        image = self.render_character(character)
        if self.transform:
            image = self.transform(image)
        char_label = self.char_labels[idx]
        jyutping_list = self.jyutpings[idx]  # Should be a list
        jyutping_str = ','.join(jyutping_list)  # Concatenate jyutpings with ','
        return image, character, char_label, jyutping_str

    def render_character(self, character):
        # FIXME: Too many hard coded values (which are probably font-specific)
        # Create a blank image with white background
        image = Image.new('L', (64, 80), color='white')
        draw = ImageDraw.Draw(image)
        # Calculate width and height of the character to center it
        width, height = textsize(text=character, font=self.font)
        # Position the character at the center of the image
        # draw.text((0, 0), character, fill='black', font=self.font)
        draw.text(((64 - width) / 2, (80 - height) / 2), character, fill='black', font=self.font)

        # Crop the image to 64x64 from the bottom - on my computer/font, the rendered
        # character is reported to be ~80pixels tall, although actually it's
        # mostly square-ish, so we need to crop away the top part
        image = image.crop((0, 16, 64, 80))

        # Distort the image a bit randomly
        image = image.rotate(np.random.randint(-10, 10), resample=Image.BICUBIC, fillcolor='white')

        # Add some noise, as random dots on the image
        image = Image.fromarray(np.where(np.random.rand(64, 64) < 0.03, 0, image))

        # Slightly blur the image
        image = image.filter(ImageFilter.GaussianBlur(radius=1))

        return image



def get_dataset():
    with open('../data/char_jyutpings/charlist_processed.json', 'r') as f:
        char_jyutpings = json.load(f)

    characters = list(char_jyutpings.keys())

    char_labels = np.array(range(len(characters)))
    jyutpings = []
    
    for char, jyutping_dict in char_jyutpings.items():
        jyutping_list = list(jyutping_dict.keys())
        jyutpings.append(jyutping_list)

    # Font path
    font_path = 'ChironHeiHK-R.ttf'

    # Transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Dataset and DataLoader
    dataset = ChineseCharacterDataset(characters, char_labels, jyutpings, font_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_char_classes = len(np.unique(char_labels))  # Adjust according to your dataset

    return (dataloader, num_char_classes)
