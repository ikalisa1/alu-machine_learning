import numpy as np
import matplotlib.pyplot as plt

# Sample data representing the number of fruit each person possesses
fruit = np.array([
    [10, 5, 8, 6],  # Farrah's apples, bananas, oranges, peaches
    [15, 3, 5, 10],  # Fred's apples, bananas, oranges, peaches
    [8, 7, 12, 4],  # Felicia's apples, bananas, oranges, peaches
])

# Define colors for each fruit
colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

# Define fruit labels and legend labels
fruits = ['Apples', 'Bananas', 'Oranges', 'Peaches']
legend_labels = ['Apples', 'Bananas', 'Oranges', 'Peaches']

# Plotting the stacked bar graph
fig, ax = plt.subplots()
bar_width = 0.5
bar_positions = np.arange(fruit.shape[0])

for i in range(fruit.shape[1]):
    ax.bar(bar_positions, fruit[:, i], bar_width, color=colors[i],
           label=legend_labels[i], bottom=np.sum(fruit[:, :i], axis=1))

# Set labels and title
ax.set_xlabel('Person')
ax.set_ylabel('Quantity of Fruit')
ax.set_title('Number of Fruit per Person')

# Set y-axis ticks and range
ax.set_yticks(np.arange(0, 81, 10))
ax.set_ylim(0, 80)

# Set x-axis ticks and labels
ax.set_xticks(bar_positions)
ax.set_xticklabels(['Farrah', 'Fred', 'Felicia'])

# Display legend
ax.legend()

# Show the plot
plt.show()
