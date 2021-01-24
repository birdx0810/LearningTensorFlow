import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(5,5))
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3))
        self.fc1 = nn.Linear(1152, 640)
        self.fc2 = nn.Linear(640, 10)
        
    def forward(self, x):
        x = F.avg_pool2d(F.tanh(self.conv1(x)), (2,2))
        x = F.avg_pool2d(F.tanh(self.conv2(x)), (2,2))
        x = flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
		
model = LeNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

def train(dataset, model, epochs):
	for epoch in range(epochs):  # loop over the dataset multiple times
		model.train() # Set model to training mode
		total_loss = 0.0
		for i, data in enumerate(dataset, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data

			# Clear out gradient before each instance
			optimizer.zero_grad()

			# forward + backward + optimize
			pred_y = model(x)               # Forward pass
			loss = criterion(pred_y, y)     # Calculate loss
			loss.backward()                 # Calculate gradient
			optimizer.step()                # Reassign learning weights

			# print statistics
			total_loss += loss.item()
			if i % 2000 == 1999:    # print every 2000 mini-batches
				print(
                    '[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, total_loss / 2000)
                )
				running_loss = 0.0

	print('Finished Training')