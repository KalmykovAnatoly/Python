import matplotlib.pyplot as plt
import torch

x_train = torch.rand(100) * 20
y_train = torch.sin(x_train)

plt.title('y = sin(x)')
plt.plot(x_train, y_train, 'o')
plt.show()

noize = torch.randn(x_train.shape) / 5.
plt.title('gaussian noise')
plt.plot(x_train, noize, 'o')
plt.show()

x_train += noize
plt.title('noisy y = sin(x)')
plt.plot(x_train, y_train, 'o')
plt.show()

x_train.unsqueeze_(1)
y_train.unsqueeze_(1)

x_validation = torch.linspace(0, 20, 100)
y_validation = (torch.sin(x_validation.data))
x_validation = x_validation.unsqueeze_(1)
y_validation = y_validation.unsqueeze_(1)


class SineNet(torch.nn.Module):

    def __init__(self, n_hidden_neurons):
        super(SineNet, self).__init__()
        self.fc1 = torch.nn.Linear(in_features=1, out_features=n_hidden_neurons, bias=True)
        self.act1 = torch.nn.Tanh()
        self.fc2 = torch.nn.Linear(in_features=n_hidden_neurons, out_features=n_hidden_neurons, bias=True)
        self.act2 = torch.nn.Tanh()
        self.fc3 = torch.nn.Linear(in_features=n_hidden_neurons, out_features=1, bias=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


sine_net = SineNet(4)


def predict(net, x, y):
    y_pred = net.forward(x)

    plt.plot(x, y, 'o')
    plt.plot(x, y_pred.data, 'o', c='r', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


predict(sine_net, x_validation, y_validation)

optimizer = torch.optim.Adam(sine_net.parameters(), lr=0.01)


def loss(pred, target):
    squares = (pred - target) ** 2
    return squares.mean()


for epoch_index in range(2000):
    optimizer.zero_grad()

    y_pred = sine_net.forward(x_train)
    loss_val = loss(y_pred, y_train)

    loss_val.backward()

    optimizer.step()

predict(sine_net, x_validation, y_validation)
