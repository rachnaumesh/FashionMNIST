import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.nn import functional as F
import gzip
import numpy as np
import os


# Load and preprocess the Fashion-MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, f'{kind}-labels-idx1-ubyte.gz')
    images_path = os.path.join(path, f'{kind}-images-idx3-ubyte.gz')

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

class CustomFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        if self.train:
            self.data, self.targets = load_mnist(self.root, kind='train')
        else:
            self.data, self.targets = load_mnist(self.root, kind='test')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = img.reshape(28, 28).astype(np.float32) / 255.0
        img = torch.from_numpy(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

# Update dataset loading
train_dataset = CustomFashionMNIST(root='./data', train=True, transform=transform)
test_dataset = CustomFashionMNIST(root='./data', train=False, transform=transform)

# Define the LeNet-5 architecture
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize the model, loss function, and optimizer
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluate on test set
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    test_loss /= len(test_loader)
    test_losses.append(test_loss)
    accuracy = 100. * correct / total
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

# Plot training and test losses
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs+1), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.show()

# Evaluate final model performance
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

final_accuracy = 100. * correct / total
print(f'Final Test Accuracy: {final_accuracy:.2f}%')

# Define the LeNet-5 architecture with dropout and batch normalization
class LeNet5Regularized(nn.Module):
    def __init__(self, dropout_rate=0.5, use_bn=False):
        super(LeNet5Regularized, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dropout = nn.Dropout(dropout_rate)
        self.use_bn = use_bn
        if use_bn:
            self.bn1 = nn.BatchNorm2d(6)
            self.bn2 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv1(x)
        if self.use_bn:
            x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2(x)
        if self.use_bn:
            x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, device):
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_accuracy = 100 * train_correct / train_total
        train_accuracies.append(train_accuracy)

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

    return train_accuracies, test_accuracies

# Hyperparameters
num_epochs = 20
learning_rate = 0.001
weight_decay = 1e-4
dropout_rate = 0.5

# Train and evaluate different models
models = {
    'No Regularization': LeNet5().to(device),
    'Dropout': LeNet5Regularized(dropout_rate=dropout_rate).to(device),
    'Weight Decay': LeNet5().to(device),
    'Batch Normalization': LeNet5Regularized(dropout_rate=0, use_bn=True).to(device)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name} model:")
    if name == 'Weight Decay':
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_accuracies, test_accuracies = train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, num_epochs, device)
    results[name] = {'train': train_accuracies, 'test': test_accuracies}

# Plotting
plt.figure(figsize=(20, 15))
for i, (name, accuracies) in enumerate(results.items()):
    plt.subplot(2, 2, i+1)
    plt.plot(range(1, num_epochs+1), accuracies['train'], label='Train')
    plt.plot(range(1, num_epochs+1), accuracies['test'], label='Test')
    plt.title(f'{name}')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

plt.tight_layout()
plt.savefig('convergence_graphs.png')
plt.close()

# Create a summary table
summary_table = {
    'Model': [],
    'Final Train Accuracy (%)': [],
    'Final Test Accuracy (%)': []
}

for name, accuracies in results.items():
    summary_table['Model'].append(name)
    summary_table['Final Train Accuracy (%)'].append(f"{accuracies['train'][-1]:.2f}")
    summary_table['Final Test Accuracy (%)'].append(f"{accuracies['test'][-1]:.2f}")

print("\nSummary Table:")
print(f"{'Model':<20} {'Final Train Accuracy (%)':<25} {'Final Test Accuracy (%)':<25}")
print("-" * 70)
for i in range(len(summary_table['Model'])):
    print(f"{summary_table['Model'][i]:<20} {summary_table['Final Train Accuracy (%)'][i]:<25} {summary_table['Final Test Accuracy (%)'][i]:<25}")

print("\nConclusions:")
print("1. No Regularization: Serves as a baseline for comparison.")
print("2. Dropout: Helps prevent overfitting by reducing interdependent learning between neurons.")
print("3. Weight Decay: Prevents overfitting by adding a penalty term to the loss function, discouraging large weights.")
print("4. Batch Normalization: Improves training stability and speed by normalizing layer inputs.")
print("\nComparison:")
print("- Dropout and Weight Decay typically show lower training accuracy but better generalization (higher test accuracy).")
print("- Batch Normalization often leads to faster convergence and can improve both training and test accuracy.")
print("- The effectiveness of each technique may vary depending on the specific dataset and model architecture.")
print("- A combination of these techniques might yield the best results in practice.")

