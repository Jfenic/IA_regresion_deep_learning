import numpy as np

def fit_linear_regression(x, y):
    """
    Fit a linear regression to the provided data.

    Parameters:
    x : array_like
        Independent values.
    y : array_like
        Dependent values.

    Returns:
    tuple
        Linear regression coefficients (slope, intercept).
    """
    x = np.array(x)
    y = np.array(y)
    
    # Calculate mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate Covariance and Variance
    cov = np.sum((x - x_mean) * (y - y_mean))
    var = np.sum((x - x_mean) ** 2)
    
    # Calculate coefficients
    slope = cov / var
    intercept = y_mean - slope * x_mean
    return slope, intercept

# Data from Anscombe's quartet
x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
y = [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]

slope, intercept = fit_linear_regression(x, y)
print(f"Slope: {slope}, Intercept: {intercept}")

# Plot generation (optional)
import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
predictions = [slope * xi + intercept for xi in x]
residuals = [y[i] - predictions[i] for i in range(len(x))]

# PLOT A: Linear Regression (Observed vs Model)
axs[0].scatter(x, y, color='blue', label='Observed Data')
axs[0].plot(x, predictions, color='red', label='Linear Regression')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
axs[0].set_title('Linear Regression Fit')
axs[0].legend()

# Plot residuals vs predictions (homoscedasticity check)
axs[1].scatter(predictions, residuals, color='green')
axs[1].axhline(y=0, color='black', linestyle='--')
axs[1].set_title('B. Residual Analysis')
axs[1].set_xlabel('Predicted Value')
axs[1].set_ylabel('Error (Residual)')

# PLOT C: Residual histogram (normality check)
axs[2].hist(residuals, bins=10, color='purple', edgecolor='black')
axs[2].set_title('C. Residual Histogram')
axs[2].set_xlabel('Error (Residual)')
axs[2].set_ylabel('Frequency')
plt.tight_layout()
plt.show()

print(f"Model: Y = {intercept:.2f} + {slope:.2f}X")

# Metric implementations

def MAE(y_true, y_pred):
    """Mean Absolute Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs(y_true - y_pred))
def MSE(y_true, y_pred):
    """Mean Squared Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean((y_true - y_pred) ** 2)
  
def RMSE(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(MSE(y_true, y_pred))

def RAE(y_true, y_pred):
    """Relative Absolute Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

def RSE(y_true, y_pred):
    """Relative Squared Error"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
  
def R2(y_true, y_pred):
    """Coefficient of Determination"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    return 1 - (ss_residual / ss_total)