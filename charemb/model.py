import torch.nn as nn

# Define the CNN architecture
class MultiTaskCNN(nn.Module):
    def __init__(self, num_char_classes, fc_width=100):
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
        self.fc1 = nn.Sequential(
            nn.Linear(128 * 8 * 8, 768),
            nn.ReLU(inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(768, fc_width),
            nn.ReLU(inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.Linear(fc_width, 768),
            nn.ReLU(inplace=True)
        )
        self.char_head = nn.Linear(768, num_char_classes)
        self.cangjie_head1 = nn.Linear(768, 27)
        self.cangjie_head2 = nn.Linear(768, 27)
        self.cangjie_head3 = nn.Linear(768, 27)
        self.cangjie_head4 = nn.Linear(768, 27)
        self.cangjie_head5 = nn.Linear(768, 27)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        char_out = self.char_head(x)
        cj1 = self.cangjie_head1(x)
        cj2 = self.cangjie_head2(x)
        cj3 = self.cangjie_head3(x)
        cj4 = self.cangjie_head4(x)
        cj5 = self.cangjie_head5(x)
        return char_out, cj1, cj2, cj3, cj4, cj5
