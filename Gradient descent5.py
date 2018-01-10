import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.axes
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#Import and normalise data
path = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
df_initial = pd.read_csv(path, delim_whitespace = True, header=None,names = ['mpg', 'cylinders', 'displacement',
                'horsepower', 'weight', 'acceleration','model_year', 'origin', 'car_name'], na_values="?")

df = df_initial.drop(['cylinders', 'horsepower', 'weight', 'model_year', 'origin', 'car_name'], axis=1)
print(df.describe())
pd.plotting.scatter_matrix(df, alpha=0.2, figsize=(7,7), diagonal='kde')
plt.show()

df_norm= (df -  df.mean()) / df.std()

df_norm.insert(0, 'Ones', 1)
df_norm = df_norm[['Ones', 'displacement', 'acceleration', 'mpg']]
print(df_norm.head())

#Define cost function
def compute_cost(X, y, theta):
    SSE = np.power(((X * theta.T) - y), 2)
    return np.sum(SSE) / (2 * len(X))

# set X (training data) and y (target variable)
X = df_norm.iloc[:,0:3]
y = df_norm.iloc[:,3:4]

print(X.head())
print(y.head())

#Convert to matrices
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0,0]))

#Define gradient descent function
def gradient_descent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:, j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = compute_cost(X, y, theta)

    return theta, cost

alpha = 0.01
iters = 10000

# perform linear regression on the data set
g, cost = gradient_descent(X, y, theta, alpha, iters)

# get the cost (error) of the model
final_cost = compute_cost(X, y, g)
print('cost:', final_cost)
print('thetas:', g)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')

#Using sklearn.linear_model
regressor = LinearRegression()
regressor.fit(X, y)
print(regressor.coef_)

#Using Statsmodels
import statsmodels.formula.api as smf
model = smf.ols(formula='mpg ~ displacement + acceleration', data=df_norm)
results_formula = model.fit()
print(results_formula.summary())

#Predicted Values
y_predict = np.matrix(X) * g.T
y_predicted = np.array(regressor.predict(X))

print("---")

#3D plot
fig = plt.figure()
ax = Axes3D(fig)

X1 = df_norm.iloc[:,1:2]
Y1 = df_norm.iloc[:,2:3]
Z = df_norm.iloc[:,3:4]

x_surf, y_surf = np.meshgrid(np.linspace(X1.min(), X1.max(), 100), np.linspace(Y1.min(), Y1.max(), 100))
x_surf1, y_surf1 = x_surf.ravel(), y_surf.ravel()
zz = g[0,0] + g[0,1] * x_surf + g[0,2] * y_surf

surf = ax.scatter(X1, Y1, Z, c='blue', marker='o', alpha=0.5)
q = ax.plot_surface(x_surf, y_surf, zz, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)

ax.set_xlabel('Displacement')
ax.set_ylabel('Acceleration')
ax.set_zlabel('MPG')
ax.set_title('3D model fit for MPG data')
plt.show()
