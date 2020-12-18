# 
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from tqdm import trange
import torch
import torchvision

# Set random seed
np.random.seed(42)
torch.manual_seed(42)

# Set Model Parameters
learning_rate = 1e-5 
epochs = 10
batch_size = 32
num_labels = 10

# Load dataset
mnist_train = torchvision.datasets.mnist('path/to/mnist_root/', train=True, download=True)
mnist_test = torchvision.datasets.mnist('path/to/mnist_root/', train=False, download=True)

train_dataloader = torch.utils.data.DataLoader(
    mnist_train,
    batch_size=batch_size,
    shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    mnist_test,
    batch_size=batch_size,
    shuffle=False
)

label_binarizer = LabelBinarizer()
label_binarizer.fit(range(num_labels))

train_onehot = label_binarizer.transform(train_y)
test_onehot = label_binarizer.transform(test_y)

class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = torch.nn.Linear(784, 10)
        torch.nn.init.ones_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.softmax = torch.nn.Softmax()
        
    def forward(self, X):
        X = torch.nn.flatten(X)
        logits = self.linear(X)
        pred = self.softmax(logits)
        
        return pred

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

start = time.time()
# Training Loop
for epoch in range(epochs):
    total_loss = 0.
    iterations = int(len(train_x)/batch_size)
    # Loop over all batches
    for i in trange(iterations):
        # Get mini batch
        batch_xs, batch_ys = get_batch(train_x, train_y, i, batch_size)
        
        batch_xs = torch.FloatTensor(batch_xs)
        batch_ys = torch.LongTensor(batch_ys)
        
        model.zero_grad()
        
        batch_ps = model(batch_xs)
        loss = criterion(batch_ps, batch_ys)
        loss.backward()
        optimizer.step()
        
        total_loss += loss
    print(f"\nEpoch:{epoch+1}\ttotal loss={total_loss/iterations}")
    
print(f"Time taken: {time.time() - start} sec")

model.eval()
test_x = torch.FloatTensor(test_x)
test_p = model(test_x)
test_p = test_p.detach().numpy().argmax(-1)

accuracy = accuracy_score(test_y, test_p)
print(f"Acc: {accuracy}")
