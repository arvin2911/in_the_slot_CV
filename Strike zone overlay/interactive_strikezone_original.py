
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def draw_pentagonal_prism(ax, center=(0, 0, 0), size=1, height=1):
    # Define the vertices of a regular pentagon
    vertices_base = np.array([
        [size * np.cos(2 * np.pi * i / 5), size * np.sin(2 * np.pi * i / 5), 0] for i in range(5)
    ])

    # Extrude the pentagon to create the sides
    vertices_top = vertices_base + [0, 0, height]
    vertices = np.vstack([vertices_base, vertices_top])

    # Create the faces of the prism
    faces = [
        [vertices_base[i], vertices_base[(i + 1) % 5], vertices_top[(i + 1) % 5], vertices_top[i]] for i in range(5)
    ]
    faces.append(vertices_base)
    faces.append(vertices_top)

    # Plot the prism
    ax.add_collection3d(Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='r', alpha=0.6))

# Create a 3D plot with interactive rotation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Call the function to draw the pentagonal prism
draw_pentagonal_prism(ax, size=1, height=2)

# Set axis limits
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([0, 3])

# Enable interactive mode
plt.ion()

# Show the plot
plt.show()

# Pause to allow user interaction
plt.pause(0.1)

# Disable interactive mode
plt.ioff()

# Display the plot
plt.show()