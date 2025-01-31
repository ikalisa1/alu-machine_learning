#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
'''This module plots all the 5 previous graphs in one figure'''

y0 = np.arange(0, 11) ** 3

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
y1 += 180

x2 = np.arange(0, 28651, 5730)
r2 = np.log(0.5)
t2 = 5730
y2 = np.exp((r2 / t2) * x2)

x3 = np.arange(0, 21000, 1000)
r3 = np.log(0.5)
t31 = 5730
t32 = 1600
y31 = np.exp((r3 / t31) * x3)
y32 = np.exp((r3 / t32) * x3)

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.figure(figsize=(10, 8)) # Creates a grid for subplots

# Plot 1 (Top left)
plt.subplot(3, 2, 1)
plt.plot(y0, 'r-')
plt.title('Plot 1: Cubic Function')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Plot 2 (Top right)
plt.subplot(3, 2, 2)
plt.scatter(x1, y1, color='magenta', alpha=0.5)
plt.title('Plot 2: Scatter Plot')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Plot 3 (Middle left)
plt.subplot(3, 2, 3)
plt.plot(x2, y2, 'b-')
plt.title('Plot 3: Exponential Decay')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.yscale('log')

# Plot 4 (Middle right)
plt.subplot(3, 2, 4)
plt.plot(x3, y31, 'r--', label='C-14')
plt.plot(x3, y32, 'g-', label='Ra-226')
plt.title('Plot 4: Two Lines')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()

# Plot 5 (Bottom, spanning two columns)
plt.subplot(3, 2, (5, 6))
plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
plt.title('Plot 5: Histogram')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Adjust layout for better spacing
plt.tight_layout()

# Title for the entire figure
plt.suptitle('All in One', fontsize='large')

# Display the plot
plt.show()
