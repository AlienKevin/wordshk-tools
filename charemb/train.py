import torch
import torch.nn as nn
import torch.optim as optim
from model import MultiTaskCNN
from dataset import get_dataset

# Set the device to use (mps device as GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Training function
def train(model, dataloader, optimizer, scheduler, criterion_char, criterion_cangjie, num_epochs=10):
    model.to(device)  # Move the model to the selected device
    model.train()
    lowest_loss = float('inf')
    best_model = None
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, characters, char_labels, jyutping_labels, cangjie_labels in dataloader:
            inputs, char_labels = inputs.to(device), char_labels.to(device)
            cangjie_labels = [label.to(device) for label in cangjie_labels]
            optimizer.zero_grad()

            # Forward pass
            char_outputs, cj1, cj2, cj3, cj4, cj5 = model(inputs)

            # Compute losses
            loss_char = criterion_char(char_outputs, char_labels)
            loss_cangjie = sum(criterion_cangjie(cj, label) for cj, label in zip([cj1, cj2, cj3, cj4, cj5], cangjie_labels))
            loss = loss_char + loss_cangjie

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}')
        if running_loss / len(dataloader) < lowest_loss:
            lowest_loss = running_loss / len(dataloader)
            best_model = model.state_dict()
            torch.save(best_model, 'charemb.pth')

# Scheduler with warmup
def get_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


if __name__ == '__main__':
    (dataloader, num_char_classes) = get_dataset(expanded_charset=True, augmented=True)

    print(f'Training with {num_char_classes} characters')

    # Initialize model, loss functions, and optimizer
    model = MultiTaskCNN(num_char_classes, fc_width=20)
    criterion_char = nn.CrossEntropyLoss()
    criterion_cangjie = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 80

    # Initialize scheduler
    num_training_steps = len(dataloader) * num_epochs
    num_warmup_steps = int(0.1 * num_training_steps)  # 10% of training steps for warmup
    scheduler = get_scheduler(optimizer, num_warmup_steps, num_training_steps)

    # Train the model
    train(model, dataloader, optimizer, scheduler, criterion_char, criterion_cangjie, num_epochs=num_epochs)
