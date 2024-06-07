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
    def __init__(self, characters, char_labels, jyutpings, font_paths, augmented):
        self.characters = characters
        self.char_labels = char_labels
        self.jyutpings = jyutpings
        self.fonts = [ImageFont.truetype(font_path, size=60) for font_path in font_paths]
        self.augmented = augmented

        # Load Cangjie labels
        self.cangjie_labels = self.load_cangjie_labels('Cangjie5_HK.txt')

        # Pre-render all character images with all fonts and apply transform if provided
        self.images = []
        for character in self.characters:
            for font in self.fonts:
                image = self.render_character(character, font)
                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5], std=[0.5])
                ])
                image = transform(image)
                self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        character_idx = idx // len(self.fonts)
        character = self.characters[character_idx]
        char_label = self.char_labels[character_idx]
        jyutping_list = self.jyutpings[character_idx]  # Should be a list
        jyutping_str = ','.join(jyutping_list)  # Concatenate jyutpings with ','
        cangjie_label = self.cangjie_labels.get(character, [0] * 5)  # Default to 0 (space) if not found
        return image, character, char_label, jyutping_str, cangjie_label

    def load_cangjie_labels(self, file_path):
        cangjie_labels = {}
        char_to_index = {chr(i + ord('a')): i + 1 for i in range(26)}
        char_to_index[' '] = 0

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    character = parts[0]
                    cangjie_code = parts[1]
                    cangjie_labels[character] = [char_to_index.get(char, 0) for char in cangjie_code.ljust(5, ' ')]
        return cangjie_labels

    def render_character(self, character, font):
        # FIXME: Too many hard coded values (which are probably font-specific)
        # Create a blank image with white background
        image = Image.new('L', (64, 80), color='white')
        draw = ImageDraw.Draw(image)
        # Calculate width and height of the character to center it
        width, height = textsize(text=character, font=font)
        # Position the character at the center of the image
        draw.text(((64 - width) / 2, (80 - height) / 2), character, fill='black', font=font)

        # Crop the image to 64x64 from the bottom - on my computer/font, the rendered
        # character is reported to be ~80pixels tall, although actually it's
        # mostly square-ish, so we need to crop away the top part
        image = image.crop((0, 16, 64, 80))

        if self.augmented:
            # Distort the image a bit randomly
            image = image.rotate(np.random.randint(-5, 5), resample=Image.BICUBIC, fillcolor='white')

            # Randomly translate the image a little bit
            max_translate = 3  # Maximum number of pixels to translate
            translate_x = np.random.randint(-max_translate, max_translate)
            translate_y = np.random.randint(-max_translate, max_translate)
            image = image.transform(
                image.size,
                Image.AFFINE,
                (1, 0, translate_x, 0, 1, translate_y),
                resample=Image.BICUBIC,
                fillcolor='white'
            )

            # Add some noise, as random dots on the image
            if np.random.rand() < 0.5:
                image = Image.fromarray(np.where(np.random.rand(64, 64) < 0.02, 0, image))

            # Slightly blur the image
            image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        return image


# FIXME: expanded charset (traditional + simplified characters) has placeholder jyutping annotations
def get_dataset(expanded_charset=False, augmented=True):
    if expanded_charset:
        with open('wordshk_chars_expanded.txt', 'r', encoding='utf-8') as file:
            characters = file.read().splitlines()
    else:
        with open('../data/char_jyutpings/charlist_processed.json', 'r') as f:
            char_jyutpings = json.load(f)
        characters = list(char_jyutpings.keys())

    char_labels = np.array(range(len(characters)))

    jyutpings = []
    if expanded_charset:
        for _ in characters:
            jyutpings.append([])
    else:
        for char, jyutping_dict in char_jyutpings.items():
            jyutping_list = list(jyutping_dict.keys())
            jyutpings.append(jyutping_list)

    # Font paths
    # if augmented:
    #     font_paths = ['ChironHeiHK-R.ttf', 'ChironHeiHK-EL.ttf']
    # else:
    font_paths = ['ChironHeiHK-R.ttf']

    # Dataset and DataLoader
    dataset = ChineseCharacterDataset(characters, char_labels, jyutpings, font_paths, augmented)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    num_char_classes = len(np.unique(char_labels))  # Adjust according to your dataset

    return (dataloader, num_char_classes)
