import torch
import torch.nn as nn
import torch.optim as optim
from model import MultiTaskCNN
from dataset import get_dataset

# Set the device to use (mps device as GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Training function
def train(model, dataloader, optimizer, criterion_char, criterion_jyutping, num_epochs=10):
    model.to(device)  # Move the model to the selected device
    model.train()
    lowest_loss = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, characters, char_labels, jyutping_labels in dataloader:
            inputs, char_labels, jyutping_labels = inputs.to(device), char_labels.to(device), jyutping_labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            char_outputs, jyutping_outputs = model(inputs)

            # Compute losses
            loss_char = criterion_char(char_outputs, char_labels)
            loss_jyutping = criterion_jyutping(jyutping_outputs, jyutping_labels.float())
            print(f'Character loss: {loss_char}')
            print(f'Jyutping loss: {loss_jyutping}')
            loss = loss_char + loss_jyutping

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
        if running_loss / len(dataloader) < lowest_loss:
            lowest_loss = running_loss / len(dataloader)
            best_model = model.state_dict()
    torch.save(best_model, 'charemb.pth')

if __name__ == '__main__':
    (dataloader, num_char_classes, num_jyutping_classes) = get_dataset()

    # Initialize model, loss functions, and optimizer
    model = MultiTaskCNN(num_char_classes, num_jyutping_classes)
    criterion_char = nn.CrossEntropyLoss()
    criterion_jyutping = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # Train the model
    train(model, dataloader, optimizer, criterion_char, criterion_jyutping, num_epochs=20)
