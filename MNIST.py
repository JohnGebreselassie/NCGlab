# Import PyTorch and other relevant libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt



device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Download and transform the MNIST dataset
transform = transforms.Compose([
    # Convert images to PyTorch tensors which also scales data from [0,255] to [0,1]
    transforms.ToTensor()
])

# Download training and test datasets
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


# Define the fully connected model - 2 layers, 28*28 -> 128, then 128-> 10
class FullyConnectedModel(nn.Module):
    def __init__(self):
        super(FullyConnectedModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)

        # '''Define the activation function for the first fully connected layer'''
        self.relu = nn.ReLU()

        # '''Define the second Linear layer to output the classification probabilities'''
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)

        # ''' Implement the rest of forward pass of the model using the layers you have defined above'''
        x = self.relu(x) #test without this line to see what it does at some point
        x = self.fc2(x)

        return x


def evaluate(model, dataloader, loss_function):
    # Evaluate model performance on the test dataset
    model.eval()
    test_loss = 0
    correct_pred = 0
    total_pred = 0
    # Disable gradient calculations when in inference mode
    with torch.no_grad():
        for images, labels in testset_loader:

            # ensure evalaution happens on the GPU

            images, labels = images.to(device), labels.to(device)

            # feed the images into the model and obtain the predictions (forward pass)
            outputs = fc_model(images)

            loss = loss_function(outputs, labels)

            # Calculate test loss
            test_loss += loss.item()*images.size(0)

           #'''make a prediction and determine whether it is correct!'''
            # identify the digit with the highest probability prediction for the images in the test dataset.
            predicted = torch.argmax(outputs, dim=1)

            # tally the number of correct predictions
            correct_pred += (predicted==labels).sum().item()

            # tally the total number of predictions
            total_pred += labels.size(0)

            #showing the guess?
            #plt.imshow(images.squeeze(), cmap=plt.cm.binary)
            #plt.show()


    # Compute average loss and accuracy
    test_loss /= total_pred
    test_acc = correct_pred / total_pred

    return test_loss, test_acc


#Train the model
def train(model, dataloader, criterion, optimizer, epochs):
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        total_loss = 0
        correct_pred = 0
        total_pred = 0

        for images, labels in trainset_loader:
            # Move tensors to GPU so compatible with model
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = fc_model(images)

            # Clear gradients before performing backward pass
            optimizer.zero_grad()
            # Calculate loss based on model predictions
            loss = loss_function(outputs, labels)
            # Backpropagate and update model parameters
            loss.backward()
            optimizer.step()

            # multiply loss by total nos. of samples in batch
            total_loss += loss.item()*images.size(0)

            # Calculate accuracy
            predicted = torch.argmax(outputs, dim=1)  # Get predicted class
            correct_pred += (predicted == labels).sum().item()  # Count correct predictions
            total_pred += labels.size(0) # Count total predictions

        # Compute metrics
        total_epoch_loss = total_loss / total_pred
        epoch_accuracy = correct_pred / total_pred
        print(f"Epoch {epoch + 1}, Loss: {total_epoch_loss}, Accuracy: {epoch_accuracy:.4f}")




# Train the model
fc_model = FullyConnectedModel().to(device) # send the model to GPU
loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(fc_model.parameters(), lr=0.1)

BATCH_SIZE = 64
trainset_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

EPOCHS = 5
train(fc_model, trainset_loader, loss_function, optimizer,EPOCHS)



#evaluate the model
testset_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loss, test_acc = evaluate(fc_model,testset_loader, loss_function)

print('Test accuracy:', test_acc)

image, label = train_dataset[0]
plt.imshow(image.squeeze(), cmap=plt.cm.binary)

plt.show()