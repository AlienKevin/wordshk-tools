import torch
import torch.nn as nn

# Define the CNN architecture
class MultiTaskCNN(nn.Module):
    def __init__(self, num_char_classes, num_jyutping_classes):
        super(MultiTaskCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True)
        )
        self.char_head = nn.Linear(256, num_char_classes)
        self.jyutping_head = nn.Linear(256, num_jyutping_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        char_out = self.char_head(x)
        jyutping_out = torch.sigmoid(self.jyutping_head(x))  # Use sigmoid for multi-label classification
        return char_out, jyutping_out
