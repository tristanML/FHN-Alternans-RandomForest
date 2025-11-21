import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- SVM CLASS IMPLEMENTATION ---

class SVM:
    """
    Support Vector Machine (SVM) implementation from scratch using Gradient Descent.
    """
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        # Hyperparameters
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        
        # Initialize weights (w) and bias (b)
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Step 1: Ensure class labels are -1 or +1 (essential for the SVM equations)
        y_ = np.where(y <= 0, -1, 1)

        # Step 2: Initialize weights and bias
        self.w = np.zeros(n_features)
        self.b = 0

        # Step 3: Gradient Descent (Training loop)
        for _ in range(self.n_iters):
            for idx, xi in enumerate(X): 
                # Check the condition: y_i * (w.x_i - b) >= 1
                condition = y_[idx] * (np.dot(xi, self.w) - self.b) >= 1

                if condition:
                    # Case 1: Correctly classified (Loss is 0, gradient is only from regularization)
                    # W update: w - lr * (2 * lambda * w)
                    self.w -= self.lr * (2 * self.lambda_param * self.w)

                else:
                    # Case 2: Misclassified or inside the margin (Loss > 0)
                    # W update: w - lr * ( (2 * lambda * w) - (y_i * x_i) )
                    self.w -= self.lr * (2 * self.lambda_param * self.w - y_[idx] * xi)
                    
                    # B update: b - lr * (-y_i)
                    self.b -= self.lr * y_[idx]

    def predict(self, X):
        # Calculate the linear function: w.x - b
        approximation = np.dot(X, self.w) - self.b
        
        # Return the sign of the approximation (+1 or -1)
        return np.sign(approximation)

# --- TESTING AND VISUALIZATION SCRIPT ---

# 1. Setup Data and Model

# Create synthetic data with 2 classes
X, y = datasets.make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1.05, random_state=42)
y = np.where(y == 0, -1, 1) # Convert {0, 1} labels to {-1, 1}

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the classifier
# Using a smaller learning rate than the default for stability in some synthetic cases
clf = SVM(learning_rate=0.0001, lambda_param=0.01, n_iters=1000)
clf.fit(X_train, y_train)

# Calculate accuracy
predictions = clf.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"SVM Classification Accuracy: {accuracy * 100:.2f}%")

# 2. Visualization Helper Function

def visualize_svm():
    
    # Helper function to solve for one variable (x2) in the hyperplane equation
    def get_hyperplane(x, w, b, offset):
        # Equation: w.x - b = offset => x2 = (-w1*x1 + b + offset) / w2
        return (-w[0] * x - b + offset) / w[1]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    
    # Scatter plot the data points
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)

    # Get min/max x values for plotting lines
    x0_min = np.amin(X[:, 0])
    x0_max = np.amax(X[:, 0])

    # Plot the three lines: Decision Boundary (0) and Margins (+1, -1)
    
    # Decision Boundary (w.x - b = 0)
    x1_0 = get_hyperplane(x0_min, clf.w, clf.b, 0)
    x1_1 = get_hyperplane(x0_max, clf.w, clf.b, 0)
    ax.plot([x0_min, x0_max], [x1_0, x1_1], 'y--') # Yellow dashed line

    # Positive Hyperplane (w.x - b = 1)
    x1_0 = get_hyperplane(x0_min, clf.w, clf.b, 1)
    x1_1 = get_hyperplane(x0_max, clf.w, clf.b, 1)
    ax.plot([x0_min, x0_max], [x1_0, x1_1], 'k') # Black solid line

    # Negative Hyperplane (w.x - b = -1)
    x1_0 = get_hyperplane(x0_min, clf.w, clf.b, -1)
    x1_1 = get_hyperplane(x0_max, clf.w, clf.b, -1)
    ax.plot([x0_min, x0_max], [x1_0, x1_1], 'k') # Black solid line

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('SVM Decision Boundary and Margins')
    plt.show()

# 3. Run Visualization

visualize_svm()