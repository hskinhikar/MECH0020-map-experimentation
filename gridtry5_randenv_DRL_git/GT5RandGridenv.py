#!/usr/bin/env python
# coding: utf-8
import sys; print(sys.executable)
# In[5]:
import sys; print(sys.executable)
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import pygame

import heapq


def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(grid, start, goal):
    start = tuple(start)
    goal = tuple(goal)

    neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # Right, Down, Left, Up
    close_set = set()
    came_from = {}
    gscore = {start: 0}
    fscore = {start: heuristic(start, goal)}
    open_heap = []

    heapq.heappush(open_heap, (fscore[start], start))
    
    while open_heap:
        current = heapq.heappop(open_heap)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            tentative_g_score = gscore[current] + heuristic(current, neighbor)
            if 0 <= neighbor[0] < len(grid) and 0 <= neighbor[1] < len(grid[0]) and grid[neighbor[0]][neighbor[1]] == 1:
                if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                    continue
                
                if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in open_heap]:
                    came_from[neighbor] = current
                    gscore[neighbor] = tentative_g_score
                    fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (fscore[neighbor], neighbor))
            else:
                continue

    return []  # No path found









# In[6]:



class RandGridWorldEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=10):
        self.size = size
        self.window_size = 512
        self.action_space = spaces.Discrete(4)  # Up, down, left, right
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=np.int64),
        })
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._grid, self._agent_location, self._target_location = self.generate_grid()
        
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

    def generate_grid(self):
        grid = np.ones((self.size, self.size), dtype=int)  # Open grid
        start, goal = self._initialize_start_target(grid)

        # Add obstacles, ensuring there is a path from start to goal
        obstacles_to_add = random.randint(1, self.size * self.size // 4)
        while obstacles_to_add > 0:
            obstacle = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if obstacle != start and obstacle != goal and grid[obstacle] == 1:
                # Temporarily add the obstacle
                grid[obstacle] = 0
                path = a_star_search(grid, start, goal)
                if not path:  # No path found, remove the obstacle
                    grid[obstacle] = 1
                else:
                    obstacles_to_add -= 1  # Obstacle successfully added

        return grid, start, goal

    def _initialize_start_target(self, grid):
        # Initialize start point
        while True:
            start = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if grid[start] == 1:  # Ensure start is not an obstacle
                break

        # Initialize goal point
        while True:
            goal = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            if grid[goal] == 1 and goal != start:  # Ensure goal is not an obstacle and not the start
                break

        return start, goal

    # Implement other methods like reset, step, render, _get_obs, _get_info, etc.
    
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self._agent_location) - np.array(self._target_location), ord=1
            )
        }

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Regenerate the grid and the start/goal locations
        self._grid, self._agent_location, self._target_location = self.generate_grid()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):
        # Map the action to the direction we walk in
        direction = self._action_to_direction[action]

        # Calculate the proposed new position based on the action
        new_position = self._agent_location + direction

        # Use np.clip to ensure the new position doesn't go out of bounds
        new_position = np.clip(new_position, 0, self.size - 1)

        # Check if the new position is an obstacle
        if self._grid[tuple(new_position)] == 1:  # Assuming 1 represents free space, 0 is an obstacle
            # Update the agent's position if new position is not an obstacle
            self._agent_location = new_position

        # Check if the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)

        # Reward is given only if the target is reached
        reward = 1 if terminated else 0

        # Get the current state observation and info about the environment
        observation = self._get_obs()
        info = self._get_info()

        # Render the environment if it's set to human mode
        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
 
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()



# In[ ]:





# In[ ]:





# In[ ]:




