# Linear_Regression_Cost_Plot
README
Visualizing Cost Function for Linear Regression
This Jupyter Notebook contains code to visualize the cost function ( J(w, b) ) for a linear regression model. The notebook includes both a 3D surface plot and a contour plot to help understand how different values of the weight (w) and bias (b) parameters affect the cost function.

Table of Contents
Introduction
Prerequisites
Installation
Usage
Code Explanation
Conclusion
License
Introduction
Linear regression is a fundamental machine learning algorithm used to model the relationship between a dependent variable and one or more independent variables. The cost function ( J(w, b) ) measures how well the model's predictions match the actual data. By visualizing the cost function, we can understand how to adjust the parameters to minimize the cost and improve the model's performance.

Prerequisites
Basic knowledge of Python programming.
Familiarity with linear regression concepts.
An internet connection to run the notebook in a cloud environment like Google Colab.
Installation
To run this notebook, you need to have Python and some essential libraries installed. You can use the following commands to install the necessary libraries:

```bash pip install numpy matplotlib Usage Open the Notebook: Use an online environment like Google Colab or Jupyter Notebook.

Copy the Code: Paste the provided code into a new notebook cell.

Run the Code: Execute the cells to generate the 3D surface plot and contour plot.

Explore: Adjust the parameters and data to see how the cost function changes.

Code Explanation

Import Libraries We import the necessary libraries for numerical calculations (numpy) and plotting (matplotlib).
python import numpy as np import matplotlib.pyplot as plt from mpl_toolkits.mplot3d import Axes3D

Define the Cost Function The compute_cost function calculates the cost for given values of w, b, X, and y.
python def compute_cost(w, b, X, y): m = X.shape[0] total_cost = 0 for i in range(m): f_wb = w * X[i] + b cost = (f_wb - y[i]) ** 2 total_cost += cost return (1 / (2 * m)) * total_cost

Generate Training Data We create a set of training data X (input data) and y (actual values).
python X = np.array([1, 2, 3, 4]) y = np.array([2, 4, 6, 8])

Create Meshgrid We generate a range of w and b values and create a meshgrid to evaluate the cost function.
python w_vals = np.linspace(-10, 10, 100) b_vals = np.linspace(-10, 10, 100) W, B = np.meshgrid(w_vals, b_vals)

Calculate Cost Values We loop through the grid to calculate the cost for each pair of w and b.
python J_vals = np.zeros(W.shape) for i in range(W.shape[0]): for j in range(W.shape[1]): J_vals[i, j] = compute_cost(W[i, j], B[i, j], X, y)

Plot the Cost Function We create both a 3D surface plot and a contour plot to visualize the cost function.
python fig = plt.figure(figsize=(14, 8))

3D surface plot
ax = fig.add_subplot(121, projection='3d') ax.plot_surface(W, B, J_vals, cmap='viridis') ax.set_xlabel('w') ax.set_ylabel('b') ax.set_zlabel('Cost') ax.set_title('3D Surface Plot of Cost Function J(w, b)')

Contour plot
ax2 = fig.add_subplot(122) contour = ax2.contour(W, B, J_vals, cmap='viridis') plt.colorbar(contour) ax2.set_xlabel('w') ax2.set_ylabel('b') ax2.set_title('Contour Plot of Cost Function J(w, b)')

plt.show() Conclusion This notebook provides a visual understanding of the cost function 𝐽 ( 𝑤 , 𝑏 ) for a linear regression model. By exploring the 3D surface plot and contour plot, you can gain insights into how different values of w and b affect the cost and how to adjust these parameters to minimize the cost.

License This project is licensed under the MIT License. Feel free to use and modify the code as needed.
