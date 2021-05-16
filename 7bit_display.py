import numpy as np
import math


def error(p, q):
    return np.linalg.norm(p - q)


Samples = np.array(
    [[1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
     [1, 1, 1, 1, 1, 0, 0, 1, 1, 1],
     [1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
     [1, 0, 1, 1, 0, 1, 1, 0, 1, 1],
     [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
     [1, 0, 0, 0, 1, 1, 1, 0, 1, 1],
     [0, 0, 1, 1, 1, 1, 1, 0, 1, 1],
     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
)  # one extra row for bias inputs

Targets = np.diag(np.ones(Samples.shape[1])) #Identity matrix
print(Samples)

N, M, P = 9, Samples.shape[0], Targets.shape[0]  # N = hidden layer neurons, M = input neurons, P = output neurons
num_samples = Samples.shape[1]
eta = 1  # Learning rate
sigma = 1  # steepness factor

V = np.random.rand(N, M) - 0.5  # Weight matrix from input to hidden; may edit this
W = np.random.rand(P, N + 1) - 0.5  # Weight matrix from hidden to output; may edit this
# print(V, W)

g = np.vectorize(lambda yin: 1 / (1 + math.exp(-sigma * yin)))  # Activation function; may be different
h = np.vectorize(lambda yin: sigma * g(yin) * (1 - g(yin)))  # Derivative of g(x); change it if g(x) is changed
np.set_printoptions(suppress=True)

# g = np.vectorize(lambda yin: (1 / math.pi) * (math.pi / 2 + math.atan(yin * sigma)))
# h = np.vectorize(lambda yin: sigma / (1 + (sigma * yin) ** 2))

print("Initial Weight matrix for hidden layer: \n", V)
print("\nInitial Weight matrix for output layer: \n", W)

n_epochs = 500
n = 0
tol = 0.1

# for _ in range(n_epochs):
while True:
    for i in range(num_samples):
        x, d = Samples[:, i].reshape(M, 1), Targets[:, i].reshape(P, 1)
        u = V @ x
        z = np.append(g(u), [[1]]).reshape((N + 1, 1))
        # print('u =', u, '\n\nz =', z)

        a = W @ z
        y = g(a)
        # print('\na =', a, '\n\ny =', y)

        Delta = (d - y) * h(a)
        delta = h(u) * (W.T[:-1] @ Delta)

        # print('\nDel =', Delta, '\n\ndel =', delta)
        # print(Delta.shape, x.shape, z.shape, u.shape, a.shape)

        Del_out = eta * (Delta @ z.T)
        del_hid = eta * (delta @ x.T)
        V = V + del_hid
        W = W + Del_out

        # print('\nV =', V, '\n\nW =', W)
        #
        # print('\ndel_hid =', del_hid)
        # print('\nDel_out =', Del_out)
        # print(h(u))

    u = V @ Samples
    z = np.vstack([g(u), np.ones(num_samples)])
    y = g(W @ z)
    if error(y, Targets) < 0.05 * np.linalg.norm(Targets):
        break
    n += 1


print("\n\nFinal Weight matrix for hidden layer: \n", V)
print("\nFinal Weight matrix for output layer: \n", W, "\n\n")
u = V @ Samples
# print(u)
z = np.vstack([g(u), np.ones(num_samples)])
y = g(W @ z)
print("The outputs corresponding to given samples are\n", y) #np.round(y))

print("\nThe number of epochs: %i & The error : %.3f\n" % (n, error(y, Targets) / np.linalg.norm(Targets)))
