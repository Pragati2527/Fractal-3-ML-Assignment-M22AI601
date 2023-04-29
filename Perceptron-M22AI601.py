import numpy as np
import matplotlib.pyplot as plt

# Sample data
X = np.array([[1, 1], [-1, -1], [0, 0.5], [0.1, 0.5], [0.2, 0.2], [0.9, 0.5]])
Y = np.array([1, -1, -1, -1, 1, 1])

# Initial weight vector and bias
w = np.array([1, 1])
b = 0

# Perceptron learning algorithm
epoch = 1
converged = False
while not converged:
    print("Epoch ", epoch, ":")
    misclassified = 0
    for i in range(len(X)):
        x = X[i]
        y = Y[i]
        y_hat = np.dot(w, x) + b
        if y_hat*y <= 0:
            w = w + y*x
            b = b + y
            misclassified += 1
            print("Sample ", i+1, ": x=", x, " y=", y, " y_hat=", y_hat, " Incorrect, Update w=", w, " b=", b)
        else:
            print("Sample ", i+1, ": x=", x, " y=", y, " y_hat=", y_hat, " Correct")
    if misclassified == 0:
        converged = True
    epoch += 1

# Final weight vector and decision boundary
print("\nFinal weight vector: w=", w, " b=", b)
print("Decision boundary: ", w[0], "x1 + ", w[1], "x2 + ", b, "= 0")

# Plot decision boundary
x = np.linspace(-1, 1, 100)
y = -w[0]/w[1]*x - b/w[1]
plt.scatter(X[:,0], X[:,1], c=Y)
plt.plot(x, y, '-r')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()