import numpy as np
import matplotlib.pyplot as plt
import random
import tkinter as tk
from tkinter import messagebox
from Qlearning import *
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# GUI Class
class MazeApp:
    def __init__(self, master):
        self.master = master
        master.title("Maze Solver")
        
        maze = load_maze_from_txt('IntroductiontoMachineLearningandNeuralNetworks\Problem8_Maze_RL\mazeArray.txt')
        self.maze = np.array(maze)
        self.start = (48, 63)  # Start position

        # 创建左侧的画布用于显示迷宫
        self.canvas_frame = tk.Frame(master)
        self.canvas_frame.pack(side=tk.LEFT)

        self.figure, self.ax = plt.subplots(figsize=(8, 5))
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack()

        # 创建右侧的输入框和按钮
        self.input_frame = tk.Frame(master)
        self.input_frame.pack(side=tk.RIGHT)

        self.label = tk.Label(self.input_frame, text="Enter goal coordinates (row, col):")
        self.label.pack()

        self.entry_row = tk.Entry(self.input_frame)
        self.entry_row.pack()
        self.entry_col = tk.Entry(self.input_frame)
        self.entry_col.pack()

        self.solve_button = tk.Button(self.input_frame, text="Solve", command=self.solve)
        self.solve_button.pack()

        self.result_label = tk.Label(self.input_frame, text="")
        self.result_label.pack()

        self.render_maze()

    def solve(self):
        try:
            goal_x = int(self.entry_row.get())
            goal_y = int(self.entry_col.get())
            
            goal = (goal_x, goal_y)
            print(goal)

            # 检查目标坐标是否在迷宫范围内
            if not (0 <= goal[0] < self.maze.shape[0] and 0 <= goal[1] < self.maze.shape[1]):
                raise ValueError("The goal coordinates are out of bounds!")
            
            # 检查目标位置是否为墙
            if self.maze[goal[0], goal[1]] == 1:
                raise ValueError("The goal is a wall!")
            # 调用 Q-learning 算法
            
            env = MazeEnv(self.maze, self.start, goal)
            q_table = q_learning(env)
            path = generate_path(env, q_table)
            print("finish q learning")
            # 调用 render 函数
            self.render_maze(path, goal)
            # 显示结果
            if path and path[-1] == goal:
                env.history = path
                env.render(goal)
            else:
                self.result_label.config(text="Cannot reach the goal.")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def render_maze(self, path=None, goal=None):
        self.ax.clear()  # 清除之前的图像
        self.ax.imshow(self.maze, cmap="hot", interpolation="nearest")
        if path:
            for step in path:
                self.ax.plot(step[1], step[0], color="red", markersize=4, marker="o")
            if goal:
                self.ax.plot(goal[1], goal[0], color="green", marker="o", markersize=6, label="Goal")

        self.ax.set_title("Maze Environment")
        self.ax.axis('on')  # 隐藏坐标轴
        self.canvas.draw()  # 刷新画布
        print("finish printing")

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.mainloop()