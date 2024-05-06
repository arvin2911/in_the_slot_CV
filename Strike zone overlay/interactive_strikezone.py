import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

def draw_pentagonal_prism(ax, zone_bounds,zone_dims_dict,center=(0, 0, 0)):
    # Define the vertices of a regular pentagon
    size = 1
    vertices_base = np.array([
         [size * np.cos(2 * np.pi * i / 5), size * np.sin(2 * np.pi * i / 5), 0] for i in range(5)
     ])
    #vertices_base = np.array([
     #zone_bounds[i][2] + zone_dims_dict.get('y_bottom') for i in range(len(zone_bounds)) 
    #])

    height = zone_dims_dict.get('y_length') #top and bottom of the zone
    # Extrude the pentagon to create the sides
    print((vertices_base[0]))
    vertices_top = vertices_base + np.array([0, 0, height])
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
if __name__ == 'main':
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Call the function to draw the pentagonal prism
    draw_pentagonal_prism(ax, size=1, height=height)
    x_lim = zone_dims_dict.get('x_length_front')
    z_lim = zone_dims_dict.get('z_length_front')+zone_dims_dict.get('z_length_back')
    # Set axis limits
    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([-height, height])
    ax.set_zlim([-z_lim, z_lim])

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
