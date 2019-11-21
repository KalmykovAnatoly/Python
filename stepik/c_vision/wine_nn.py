import random

import numpy as np
import sklearn.datasets
import torch
from sklearn.model_selection import train_test_split

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

wine = sklearn.datasets.load_wine()

X_train, X_test, y_train, y_test = train_test_split(
    wine.data[:, 0:2],
    wine.target,
    test_size=0.3,
    shuffle=True
)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


class WineNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(WineNet, self).__init__()

        self.fc1 = torch.nn.Linear(2, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()

        self.fc2 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons)
        self.act2 = torch.nn.Sigmoid()

        self.fc3 = torch.nn.Linear(n_hidden_neurons, 3)
        self.sm = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


wine_net = WineNet(1)

loss = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(wine_net.parameters(), lr=0.001)

np.random.permutation(5)

batch_size = 10

for epoch in range(10000):
    order = np.random.permutation(len(X_train))
    for start_index in range(0, len(X_train), batch_size):
        optimizer.zero_grad()

        batch_indexes = order[start_index:start_index + batch_size]

        x_batch = X_train[batch_indexes]
        y_batch = y_train[batch_indexes]

        predictions = wine_net.forward(x_batch)

        loss_val = loss(predictions, y_batch)
        loss_val.backward()

        optimizer.step()

    if epoch % 100 == 0:
        test_predictions = wine_net.forward(X_test)
        test_predictions = test_predictions.argmax(dim=1)
        print((test_predictions == y_test).float().mean())
