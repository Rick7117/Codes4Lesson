import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import messagebox
from Qlearning import *


# Example usage
if __name__ == "__main__":
    # Define maze (0 = free, 1 = wall)
    maze = load_maze_from_txt('Introduction to Machine Learning and Neural Networks\Problem8_Maze_RL\mazeArray.txt')
    print(len(maze), len(maze[0]))
    maze = np.array(maze)

    start = (1, 1)
    goal = (48, 63)

    env = MazeEnv(maze, start, goal)
    q_table = q_learning(env, episodes=5000)

    path = generate_path(env, q_table)

    # Visualize the path
    print("Path taken:", path)
    for step in path:
        env.current_position = step
        env.render()