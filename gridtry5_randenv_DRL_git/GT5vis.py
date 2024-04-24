#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np

def visualize(grid, path, start, goal):
    grid_array = np.array(grid)
    fig, ax = plt.subplots()
    
    # Use 'origin=lower' to ensure that (0,0) is at the top-left
    ax.imshow(grid_array, cmap=plt.cm.Dark2, origin='upper')

    # Swap the start and goal coordinates for plotting
    ax.plot(start[1], start[0], 'go', markersize=10, label="Start")  # Plot (column, row) instead of (row, column)
    ax.plot(goal[1], goal[0], 'r*', markersize=15, label="Goal")

    # Swap the path coordinates for plotting
    for point in path:
        ax.plot(point[1], point[0], 'b.', markersize=8)  # Plot (column, row) for each point in path

    # Draw grid lines
    for x in range(len(grid_array[0])):  # Columns
        ax.axvline(x - 0.5, lw=2, color='k', zorder=5)
    for y in range(len(grid_array)):  # Rows
        ax.axhline(y - 0.5, lw=2, color='k', zorder=5)

    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    
    # Set the axis limits to include all squares
    ax.set_xlim(-0.5, len(grid_array[0]) - 0.5)
    ax.set_ylim(-0.5, len(grid_array) - 0.5)

    # Ensure the aspect ratio is equal to make the cells square
    ax.set_aspect('equal')

    # Invert y-axis to have (0,0) at the top-left
    ax.invert_yaxis()

    plt.legend()
    plt.show()

# Call the visualize function with your grid, path, start, and goal
# visualize(grid, path, start, goal)


# In[14]:


import matplotlib.pyplot as plt
import numpy as np

def visualize1(grid, path, start, goal):
    grid_array = np.array(grid)
    fig, ax = plt.subplots()
    
    ax.imshow(grid_array, cmap=plt.cm.Dark2, origin='upper')

    # Plot the start and goal coordinates
    ax.plot(start[1], start[0], 'go', markersize=10, label="Start")
    ax.plot(goal[1], goal[0], 'r*', markersize=15, label="Goal")

    # Convert path to x and y coordinates for plotting
    x_coords, y_coords = zip(*path)
    ax.plot(y_coords, x_coords, color='blue', linewidth=2)  # Connect points with a line

    # Mark each point on the path
    for point in path:
        ax.plot(point[1], point[0], 'b.', markersize=8)

    # Draw grid lines
    for x in range(len(grid_array[0])):
        ax.axvline(x - 0.5, lw=2, color='k', zorder=5)
    for y in range(len(grid_array)):
        ax.axhline(y - 0.5, lw=2, color='k', zorder=5)

    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    ax.set_xlim(-0.5, len(grid_array[0]) - 0.5)
    ax.set_ylim(-0.5, len(grid_array) - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.legend()
    plt.show()

# Example usage
# visualize(grid, path, start, goal)


# In[15]:


##visualisation for dijkstra


import matplotlib.pyplot as plt
import numpy as np

def visualize1(grid, path, start, goal):
    grid_array = np.array(grid)
    fig, ax = plt.subplots()
    ax.imshow(grid_array, cmap=plt.cm.Dark2, origin='upper')
    ax.plot(start[1], start[0], 'go', markersize=10, label="Start")
    ax.plot(goal[1], goal[0], 'r*', markersize=15, label="Goal")
    x_coords, y_coords = zip(*path)
    ax.plot(y_coords, x_coords, color='blue', linewidth=2)  # Path
    for point in path:
        ax.plot(point[1], point[0], 'b.', markersize=8)
    for x in range(len(grid_array[0])):
        ax.axvline(x - 0.5, lw=2, color='k', zorder=5)
    for y in range(len(grid_array)):
        ax.axhline(y - 0.5, lw=2, color='k', zorder=5)
    ax.set_xlabel('Columns')
    ax.set_ylabel('Rows')
    ax.set_xlim(-0.5, len(grid_array[0]) - 0.5)
    ax.set_ylim(-0.5, len(grid_array) - 0.5)
    ax.set_aspect('equal')
    ax.invert_yaxis()
    plt.legend()
    plt.show()




# In[16]:


def visualize4(grid, path, start, goal):
    grid_array = np.array(grid)
    fig, ax = plt.subplots()
    
    # Invert the grid for proper visualization: obstacles as 1, free path as 0
    inverted_grid = np.where(grid_array == 1, 0, 1)
    
    # Visualize the grid
    ax.imshow(inverted_grid, cmap=plt.cm.gray, origin='lower')

    # Plot the start and goal
    ax.plot(start[1], start[0], 'go', markersize=10, label="Start")  # (x, y)
    ax.plot(goal[1], goal[0], 'r*', markersize=15, label="Goal")     # (x, y)

    # Ensure the start position is at the beginning of the path list
    if path and path[0] != start:
        path.insert(0, start)

    # Unpack and plot the path coordinates, ensuring they are continuous
    if path:
        y_coords, x_coords = zip(*path)  # swap x and y for plotting
        ax.plot(x_coords, y_coords, color='blue', linewidth=2, label="Path")

    # Add grid, legend, and labels
    #ax.grid(True)
    ax.legend()
    #ax.set_xlabel("X-axis")
    #ax.set_ylabel("Y-axis")
    plt.show()


# In[17]:


import matplotlib.pyplot as plt
import numpy as np
def visualize5(grid, path, start, goal):
    grid_array = np.array(grid)
    fig, ax = plt.subplots()
    
    # Convert start and goal to NumPy arrays for consistent comparison
    start_array = np.array(start)
    goal_array = np.array(goal)
    
    # Invert the grid for proper visualization: obstacles as 1, free path as 0
    inverted_grid = np.where(grid_array == 1, 0, 1)
    
    # Visualize the grid
    ax.imshow(inverted_grid, cmap=plt.cm.gray, origin='lower')

    # Plot the start and goal
    ax.plot(start_array[1], start_array[0], 'go', markersize=10, label="Start")  # (x, y)
    ax.plot(goal_array[1], goal_array[0], 'r*', markersize=15, label="Goal")     # (x, y)

    # Ensure the start position is at the beginning of the path list
    if path and not np.array_equal(path[0], start_array):
        path.insert(0, start)

    # Unpack and plot the path coordinates, ensuring they are continuous
    if path:
        y_coords, x_coords = zip(*path)  # swap x and y for plotting
        ax.plot(x_coords, y_coords, color='blue', linewidth=2, label="Path")

    # Remove grid, ticks, and labels
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()

    # Hide axis labels
    ax.set_xlabel('')
    ax.set_ylabel('')

    plt.show()


# In[18]:


import matplotlib.pyplot as plt

def visualize6(grid, path, start, goal):
    fig, ax = plt.subplots()
    # Assuming 0 is an obstacle and 1 is a free path
    ax.imshow(grid, cmap='gray', interpolation='none')
    
    # Convert path into x and y coordinates for plotting
    xs, ys = zip(*path)
    
    ax.plot(xs, ys, color='blue', linewidth=2, label='Agent Path')  # Path
    ax.plot(*start, 'go', label='Start')  # Start position
    ax.plot(*goal, 'ro', label='Goal')  # Goal position
    
    ax.legend()
    plt.show()


# In[19]:


import numpy as np

def visualize7(grid, path, start, goal):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')  # Assuming obstacles are 0 (black) and free space is 1 (white)

    # Extract x and y coordinates from the path
    y_coords, x_coords = zip(*path)
    
    # Plot the path
    ax.plot(x_coords, y_coords, color='blue', linewidth=2, label='Path')

    # Plot the start and goal
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')  # Green for start
    ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')    # Red for goal

    ax.legend()
    plt.show()


# In[20]:


import numpy as np
import matplotlib.pyplot as plt

def visualize8(grid, path, start, goal):
    fig, ax = plt.subplots()
    
    # Invert the grid so that obstacles are white (1) and free space is black (0)
    inverted_grid = np.where(grid == 0, 1, 0)
    
    ax.imshow(inverted_grid, cmap='gray')  # Now obstacles are white and free space is black

    # Extract x and y coordinates from the path
    y_coords, x_coords = zip(*path)
    
    # Plot the path
    ax.plot(x_coords, y_coords, color='blue', linewidth=2, label='Path')

    # Plot the start and goal
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')  # Green for start
    ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')    # Red for goal

    ax.legend()
    plt.show()


# In[21]:


#vis7 but path connected to start point

def visualize9(grid, path, start, goal):
    fig, ax = plt.subplots()
    ax.imshow(grid, cmap='gray')  # Assuming obstacles are 0 (black) and free space is 1 (white)

    # Ensure the start point is the first element of the path
    if not np.array_equal(path[0], start):
        path.insert(0, start)

    # Ensure the goal point is the last element of the path
    if not np.array_equal(path[-1], goal):
        path.append(goal)

    # Plot the start and goal
    ax.plot(start[1], start[0], 'go', markersize=10, label='Start')  # Green for start
    ax.plot(goal[1], goal[0], 'ro', markersize=10, label='Goal')    # Red for goal

    # Plot the path with checks for obstacles
    for i in range(len(path)-1):
        point_a = path[i]
        point_b = path[i+1]
        
        # Check if the path is moving horizontally or vertically
        if point_a[0] == point_b[0] or point_a[1] == point_b[1]:
            # Check for obstacles in the grid. This assumes that the grid cells with obstacles are marked with 0.
            y_min, y_max = sorted([point_a[0], point_b[0]])
            x_min, x_max = sorted([point_a[1], point_b[1]])
            if np.all(grid[y_min:y_max+1, x_min:x_max+1] != 0):
                ax.plot([point_a[1], point_b[1]], [point_a[0], point_b[0]], color='blue', linewidth=2)

    ax.legend()
    plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




