import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torch.onnx
import torch.nn.functional as F

# load EMNIST dataset
emnist_data = datasets.EMNIST(
    './EMNIST',
    split='letters',
    # split='balanced',
    train=True, download=True,
    transform=transforms.ToTensor())

data_loader = torch.utils.data.DataLoader(emnist_data, batch_size=2, shuffle=True)

print("EMNIST dataset loaded")


# create and train model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 64)
        self.dropout = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 47)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


model = Net()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
criterion = nn.NLLLoss()

print("Model created")

for epoch in range(10):
    for data, target in data_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
#
# convert to ONNX format
print("Converting to ONNX format")
torch.onnx.export(model, torch.randn(1, 1, 28, 28), "model.onnx")
