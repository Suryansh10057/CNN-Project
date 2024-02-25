import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torchsummary import summary

# Define CIFAR-10 data loading and preprocessing
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

batch_size = 256
num_epochs = 50
learning_rate = 0.001

# Load CIFAR-10 dataset
train_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10/train", train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR10(root="./CIFAR10/test", train=False, transform=transform, download=True)

train_loader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

# Define CNN-Vanilla
class CNNVanilla(nn.Module):
    def __init__(self):
        super(CNNVanilla, self).__init__()
        # Define architecture similar to CNN-Resnet without residual connections
        # Modify this architecture to match CNN-Resnet's structure
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)  # Output has 10 classes for CIFAR-10
        )

    def forward(self, x):
        return self.model(x)


class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()  # Identity mapping for the residual connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += self.shortcut(x)  # Residual connection
        out = self.relu(out)
        return out

class CNNResnet(nn.Module):
    def __init__(self, num_fc_layers=2):
        super(CNNResnet, self).__init__()
        self.num_fc_layers = num_fc_layers

        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 64, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )

        # Calculate the number of features after convolutional layers
        num_features = 64 * 2 * 2 

        # Additional fully connected layers based on num_fc_layers
        fc_layers = []
        for _ in range(num_fc_layers):
            fc_layers.append(nn.Linear(num_features, num_features // 2))
            fc_layers.append(nn.ReLU())
            num_features //= 2
        self.fc_layers = nn.Sequential(*fc_layers)
        self.output_layer = nn.Linear(num_features, 10)  # Output has 10 classes for CIFAR-10

    def forward(self, x):
        x = self.model(x)
        x = self.fc_layers(x)
        x = self.output_layer(x)
        return x

# Create CNNResnet with a different number of fully connected layers
cnn_resnet_exp4a = CNNResnet(num_fc_layers=2)  # Modify the number of fully connected layers as needed
cnn_resnet_exp4b = CNNResnet(num_fc_layers=4)  # Modify the number of fully connected layers as needed

# Select the appropriate training device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
cnn_vanilla = CNNVanilla().to(device)
cnn_resnet = CNNResnet().to(device)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer_vanilla = optim.Adam(cnn_vanilla.parameters(), lr=learning_rate)
optimizer_resnet = optim.Adam(cnn_resnet.parameters(), lr=learning_rate)

# Training the CNN-Vanilla model
cnn_vanilla.train()
for epoch in range(num_epochs):
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_vanilla.zero_grad()
        outputs = cnn_vanilla(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_vanilla.step()
        train_loss += loss.item()

    print(f'CNN-Vanilla - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader)}')

# Training the CNN-Resnet model
cnn_resnet.train()
for epoch in range(num_epochs):
    train_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_resnet.zero_grad()
        outputs = cnn_resnet(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_resnet.step()
        train_loss += loss.item()

    print(f'CNN-Resnet - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader)}')

# Evaluate the models on the test set
cnn_vanilla.eval()
cnn_resnet.eval()

correct_vanilla = 0
correct_resnet = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs_vanilla = cnn_vanilla(images)
        outputs_resnet = cnn_resnet(images)

        _, predicted_vanilla = torch.max(outputs_vanilla, 1)
        _, predicted_resnet = torch.max(outputs_resnet, 1)

        total += labels.size(0)
        correct_vanilla += (predicted_vanilla == labels).sum().item()
        correct_resnet += (predicted_resnet == labels).sum().item()

accuracy_vanilla = 100 * correct_vanilla / total
accuracy_resnet = 100 * correct_resnet / total

print(f'CNN-Vanilla Test Accuracy: {accuracy_vanilla}%')
print(f'CNN-Resnet Test Accuracy: {accuracy_resnet}%')

# Model summary
print("CNN-Vanilla Model Summary:")
summary(cnn_vanilla, (3, 32, 32))

print("CNN-Resnet Model Summary:")
summary(cnn_resnet, (3, 32, 32))

# Experiment 1: Report and compare the performances of [CNN-Vanilla] and [CNN-Resnet].
# Also, compare the number of parameters of the two networks.
print("Experiment 1: Comparing CNN-Vanilla and CNN-Resnet")
total_parameters_vanilla = sum(p.numel() for p in cnn_vanilla.parameters())
total_parameters_resnet = sum(p.numel() for p in cnn_resnet.parameters())

print(f"Number of Parameters (CNN-Vanilla): {total_parameters_vanilla}")
print(f"Number of Parameters (CNN-Resnet): {total_parameters_resnet}")

print(f"Test Accuracy (CNN-Vanilla): {accuracy_vanilla}%")
print(f"Test Accuracy (CNN-Resnet): {accuracy_resnet}%")

# Experiment 2: Study the Effect of Data Normalization
print("Experiment 2: Studying the Effect of Data Normalization")

# Define a new CNN model with the best architecture from Experiment 1
best_model = cnn_vanilla if accuracy_vanilla > accuracy_resnet else cnn_resnet

# Define two sets of transformations with and without data normalization
transform_with_normalization = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

transform_without_normalization = transforms.Compose([
    transforms.ToTensor(),
])

# Load CIFAR-10 dataset with data normalization
train_dataset_with_norm = torchvision.datasets.CIFAR10(root="./CIFAR10/train", train=True, transform=transform_with_normalization, download=True)
test_dataset_with_norm = torchvision.datasets.CIFAR10(root="./CIFAR10/test", train=False, transform=transform_with_normalization, download=True)

# Load CIFAR-10 dataset without data normalization
train_dataset_no_norm = torchvision.datasets.CIFAR10(root="./CIFAR10/train", train=True, transform=transform_without_normalization, download=True)
test_dataset_no_norm = torchvision.datasets.CIFAR10(root="./CIFAR10/test", train=False, transform=transform_without_normalization, download=True)

# Create data loaders for both cases
train_loader_with_norm = torch.utils.data.DataLoader(train_dataset_with_norm, shuffle=True, batch_size=batch_size)
test_loader_with_norm = torch.utils.data.DataLoader(test_dataset_with_norm, batch_size=batch_size)

train_loader_no_norm = torch.utils.data.DataLoader(train_dataset_no_norm, shuffle=True, batch_size=batch_size)
test_loader_no_norm = torch.utils.data.DataLoader(test_dataset_no_norm, batch_size=batch_size)

# Training the best model with data normalization
best_model.train()
optimizer_best = optim.Adam(best_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_loss = 0
    for images, labels in train_loader_with_norm:
        images, labels = images.to(device), labels.to(device)
        optimizer_best.zero_grad()
        outputs = best_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_best.step()
        train_loss += loss.item()

    print(f'Best Model with Data Normalization - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader_with_norm)}')

# Evaluate the best model with data normalization on the test set
best_model.eval()
correct_best_with_norm = 0
total_best_with_norm = 0

with torch.no_grad():
    for images, labels in test_loader_with_norm:
        images, labels = images.to(device), labels.to(device)
        outputs_best_with_norm = best_model(images)
        _, predicted_best_with_norm = torch.max(outputs_best_with_norm, 1)
        total_best_with_norm += labels.size(0)
        correct_best_with_norm += (predicted_best_with_norm == labels).sum().item()

accuracy_vanilla_with_norm = 100 * correct_best_with_norm / total_best_with_norm
print(f'Best Model with Data Normalization Test Accuracy: {accuracy_vanilla_with_norm}%')

# Training the best model without data normalization
best_model.train()
optimizer_best = optim.Adam(best_model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_loss = 0
    for images, labels in train_loader_no_norm:
        images, labels = images.to(device), labels.to(device)
        optimizer_best.zero_grad()
        outputs = best_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_best.step()
        train_loss += loss.item()

    print(f'Best Model without Data Normalization - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader_no_norm)}')

# Evaluate the best model without data normalization on the test set
best_model.eval()
correct_best_no_norm = 0
total_best_no_norm = 0

with torch.no_grad():
    for images, labels in test_loader_no_norm:
        images, labels = images.to(device), labels.to(device)
        outputs_best_no_norm = best_model(images)
        _, predicted_best_no_norm = torch.max(outputs_best_no_norm, 1)
        total_best_no_norm += labels.size(0)
        correct_best_no_norm += (predicted_best_no_norm == labels).sum().item()

accuracy_resnet_with_norm = 100 * correct_best_no_norm / total_best_no_norm
print(f'Best Model without Data Normalization Test Accuracy: {accuracy_resnet_with_norm}%')


# Evaluate the best model without data normalization on the test set
best_model.eval()
correct_best_no_norm = 0
total_best_no_norm = 0

with torch.no_grad():
    for images, labels in test_loader_no_norm:
        images, labels = images.to(device), labels.to(device)
        outputs_best_no_norm = best_model(images)
        _, predicted_best_no_norm = torch.max(outputs_best_no_norm, 1)
        total_best_no_norm += labels.size(0)
        correct_best_no_norm += (predicted_best_no_norm == labels).sum().item()

accuracy_best_no_norm = 100 * correct_best_no_norm / total_best_no_norm
print(f'Best Model without Data Normalization Test Accuracy: {accuracy_best_no_norm}%')

# Experiment 3: Study the Effect of Different Optimizers
print("Experiment 3: Study the Effect of Different Optimizers")

# Define a new CNN model with the best architecture from Experiment 1
best_model = cnn_vanilla if accuracy_vanilla > accuracy_resnet else cnn_resnet

# Define a list of optimizers to be tested
optimizers = [
    {'name': 'Stochastic Gradient Descent', 'optimizer': 'SGD'},
    {'name': 'Mini-batch Gradient Descent (No Momentum)', 'optimizer': 'SGD', 'momentum': 0},
    {'name': 'Mini-batch Gradient Descent (Momentum 0.9)', 'optimizer': 'SGD', 'momentum': 0.9},
    {'name': 'Adam Optimizer', 'optimizer': 'Adam'}
]

# Training and evaluating the best model with different optimizers
for optimizer_info in optimizers:
    optimizer_name = optimizer_info['name']
    optimizer_type = optimizer_info['optimizer']


    optimizer_type = optimizer_info.get('type', 'Adam')  # Default to 'Adam' if 'type' key doesn't exist
    learning_rate = optimizer_info.get('lr', 0.001)  # Default learning rate to 0.001 if 'lr' key doesn't exist

if optimizer_type == 'SGD':
    momentum = optimizer_info.get('momentum', 0.9)  # Default momentum to 0.9 if 'momentum' key doesn't exist
    optimizer = optim.SGD(best_model.parameters(), lr=learning_rate, momentum=momentum)
else:
    optimizer = optim.Adam(best_model.parameters(), lr=learning_rate)


    best_model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for images, labels in train_loader_with_norm:  # Using data normalization
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = best_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f'Best Model with {optimizer_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader_with_norm)}')

    # Evaluate the best model with different optimizers on the test set
    best_model.eval()
    correct_best_with_optimizer = 0
    total_best_with_optimizer = 0

    with torch.no_grad():
        for images, labels in test_loader_with_norm:  # Using data normalization
            images, labels = images.to(device), labels.to(device)
            outputs_best_with_optimizer = best_model(images)
            _, predicted_best_with_optimizer = torch.max(outputs_best_with_optimizer, 1)
            total_best_with_optimizer += labels.size(0)
            correct_best_with_optimizer += (predicted_best_with_optimizer == labels).sum().item()

    accuracy_best_with_optimizer = 100 * correct_best_with_optimizer / total_best_with_optimizer
    print(f'Best Model with {optimizer_name} Test Accuracy: {accuracy_best_with_optimizer}%')



# Experiment 4: Study the Effect of Network Depth
print("Experiment 4: Study the Effect of Network Depth")

# Define a new CNN model with the best architecture from Experiment 2
best_model_experiment2 = cnn_vanilla if accuracy_vanilla_with_norm > accuracy_resnet_with_norm else cnn_resnet

# Define three different model architectures for Experiment 4
# a) Four level Resnet block with two fully-connected layers
# b) Three level Resnet blocks with four fully-connected layers
# c) Original model with three level Resnet block with two fully-connected layers

class FourLevelResNet(nn.Module):
    def __init__(self):
        super(FourLevelResNet, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 64, stride=2),
            ResNetBlock(64, 64, stride=2),  # Additional block
            ResNetBlock(64, 64, stride=2),  # Additional block
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

class ThreeLevelResNetWithFourFC(nn.Module):
    def __init__(self):
        super(ThreeLevelResNetWithFourFC, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            ResNetBlock(32, 64, stride=2),
            ResNetBlock(64, 64, stride=2),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 512),
            nn.ReLU(),
            nn.Linear(512, 512),  # Additional fully connected layer
            nn.ReLU(),
            nn.Linear(512, 512),  # Additional fully connected layer
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        return self.model(x)

# Define the architectures for Experiment 4
four_level_resnet = FourLevelResNet().to(device)
three_level_resnet_with_four_fc = ThreeLevelResNetWithFourFC().to(device)

# Define the optimizers
optimizer_four_level_resnet = optim.Adam(four_level_resnet.parameters(), lr=learning_rate)
optimizer_three_level_resnet_with_four_fc = optim.Adam(three_level_resnet_with_four_fc.parameters(), lr=learning_rate)

# Training and evaluating the models for Experiment 4
for model, model_name, optimizer in [
    (best_model_experiment2, 'Best Model from Experiment 2', optim.Adam(best_model_experiment2.parameters(), lr=learning_rate)),
    (four_level_resnet, 'Four Level Resnet', optimizer_four_level_resnet),
    (three_level_resnet_with_four_fc, 'Three Level Resnet with Four FC Layers', optimizer_three_level_resnet_with_four_fc)
]:
    model.train()
    for epoch in range(num_epochs):
        train_loss = 0
        for images, labels in train_loader_with_norm:  # Using data normalization
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

    print(f'{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader_with_norm)}')

    # Evaluate the models for Experiment 4 on the test set
    model.eval()
    correct_model = 0
    total_model = 0

    with torch.no_grad():
        for images, labels in test_loader_with_norm:  # Using data normalization
            images, labels = images.to(device), labels.to(device)
            outputs_model = model(images)
            _, predicted_model = torch.max(outputs_model, 1)
            total_model += labels.size(0)
            correct_model += (predicted_model == labels).sum().item()

    accuracy_model = 100 * correct_model / total_model
    print(f'{model_name} Test Accuracy: {accuracy_model}%')
