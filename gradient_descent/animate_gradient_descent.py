# animate_gradient_descent.py: From the following blog posting on ainimating gradient descent. 
#
# https://medium.com/@sdmuhsin3011/master-gradient-descent-through-visual-animation-8d2858afefa8
#


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

def function(x):
    return x**4 - 4*x**3 + 0.5*x**2 + 10 * np.cos(x)

def derivative(x):
    return 4*x**3 - 12*x**2 + x - 10 * np.sin(x)

def gradient_descent(starting_point, learning_rate, n_iterations):
    x = starting_point
    x_history = [x]
    for _ in range(n_iterations):
        grad = derivative(x)
        x = x - learning_rate * grad
        x_history.append(x)
    return x_history


# Settings for the animation
starting_point = -1
learning_rate = 0.01
n_iterations = 50
points = gradient_descent(starting_point, learning_rate, n_iterations)

# Create a figure and axes
fig, ax = plt.subplots()
sns.set(style="whitegrid")

# Generate data for the function's plot
x = np.linspace(-2, 4, 400)
y = function(x)
sns.lineplot(x=x, y=y, ax=ax, label='f(x) = x^4 - 4x^3 + 4x^2', color='blue')

# Initialize the plot with the first point
line, = ax.plot([], [], 'ro-', label='Gradient Steps')
ax.legend()

def init():
    line.set_data([], [])
    return line,

def update(frame):
    xdata = [points[i] for i in range(frame + 1)]
    ydata = [function(x) for x in xdata]
    line.set_data(xdata, ydata)
    return line,

# Create and save the animation
ani = FuncAnimation(fig, update, frames=np.arange(0, len(points)),
                    init_func=init, blit=True)
ani.save('gradient_descent.gif', writer='pillow', fps=2)

plt.title('Animated Gradient Descent Visualization')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()
