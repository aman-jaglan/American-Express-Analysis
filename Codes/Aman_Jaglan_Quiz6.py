#Author: Aman Jaglan
#GWID: G45030269

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(input):
    result = 1.0/(1.0+np.exp(-(input)))
    return result
def derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))
def g_final(x):
    return np.exp(-1*np.abs(x)) * np.sin(np.pi * x)


def model(w1, w2, b1, b2, p, t):
    predicted = []
    for i, input_val in enumerate(p):
        n1 = w1 * input_val + b1
        a1 = sigmoid(n1)
        n2 = np.dot(w2, a1) + b2
        a2 = sigmoid(n2)
        predicted.append(a2.item())

    sorted_indices = sorted(range(len(p)), key=lambda k: p[k])
    sorted_p = [p[i] for i in sorted_indices]
    sorted_t = [t[i] for i in sorted_indices]
    sorted_predicted = [predicted[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.plot(sorted_p, sorted_t, label='True Output', color='black')
    plt.plot(sorted_p, sorted_predicted, label='Network Predictions', linestyle='--', color='blue')
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.title('Comparison of Neural Network Outputs')
    plt.show()

def batch(p, t, n_neurons, alpha, epochs):
    w1 = np.array([[1] * n_neurons]).T
    w2 = np.array([[1] * n_neurons])
    b1 = np.array([[-1] * n_neurons]).T
    b2 = 1

    Error_epoch = []
    e = 0

    for epoch in range(epochs):
        e_total = 0
        sn_avg = []
        sm_avg = []
        for i, input_val in enumerate(p):
            n1 = w1 * input_val + b1
            a1 = sigmoid(n1)
            n2 = np.dot(w2, a1) + b2
            a2 = sigmoid(n2)
            e = t[i] - a2
            sn = -2 * derivative(n2) * e
            sm = derivative(n1) * (sn * w2.T)
            sn_avg.append(sn)
            sm_avg.append(sm)
            e_total += (e ** 2).sum()

        sn = np.mean(sn_avg)
        sm = np.mean(sm_avg)
        w2 = w2 - alpha * sn * a1.T
        w1 = w1 - alpha * sm * input_val
        b2 = b2 - alpha * sn
        b1 = b1 - alpha * sm
        mse = e_total / len(p)
        Error_epoch.append(mse)

    return w1, b1, w2, b2, Error_epoch

def SGD(p, t, n_neurons, alpha, epochs):
    w1 = np.random.rand(n_neurons, 1) - 0.5
    w2 = np.random.rand(1, n_neurons) - 0.5
    b1 = np.random.rand(n_neurons, 1) - 0.5
    b2 = np.random.rand(1) - 0.5
    Error_epoch = []
    e=0

    for epoch in range(epochs):
        e_total = 0
        for i, input_val in enumerate(p):
            n1 = w1 * input_val + b1
            a1 = sigmoid(n1)
            n2 = np.dot(w2, a1) + b2
            a2 = sigmoid(n2)
            e = t[i] - a2
            sn = -2 * derivative(n2) * e
            sm = derivative(n1) * (sn * w2.T)

            w2 = w2-alpha * sn * a1.T
            w1 = w1-alpha * sm * input_val
            b2 = b2-alpha * sn
            b1 = b1-alpha * sm


            e_total += (e ** 2).sum()

        mse = e_total / len(p)
        Error_epoch.append(mse)

    return w1, b1, w2, b2, Error_epoch
p = np.random.uniform(1, 10, 500)
t = [g_final(i) for i in p]

#%%
# Training the model
w1, b1, w2, b2, Error_epoch = SGD(p, t, 2, 0.1, 100)
values = [value.item() for array in Error_epoch for value in array.flatten()]
plt.plot(values)
plt.title('Error Progression over Epochs for SGD')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.show()

values = [value.item() for array in Error_epoch for value in array.flatten()]
log_values = np.log([value if value > 0 else 1e-10 for value in values])
plt.plot(log_values)
plt.title('Error Progression over Epochs for SGD over Logscale')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.show()

model(w1,w2,b1,b2,p,t)


#%%
w1, b1, w2, b2, Error_epoch = batch(p, t, 2, 0.1, 100)

values = [value.item() for array in Error_epoch for value in array.flatten()]
plt.plot(values)
plt.title('Error Progression over Epochs for BGD')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.show()

values = [value.item() for array in Error_epoch for value in array.flatten()]
log_values = np.log([value if value > 0 else 1e-10 for value in values])
plt.plot(log_values)
plt.title('Error Progression over Epochs for BGD over Logscale')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.grid(True)
plt.show()

model(w1,w2,b1,b2,p,t)

#%%





