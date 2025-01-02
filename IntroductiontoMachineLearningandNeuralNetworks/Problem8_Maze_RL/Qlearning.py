import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import sys, os
import random

class MazeEnv:
    def __init__(self, maze_array, start, goal):
        self.maze = maze_array
        self.start = start
        self.goal = goal
        self.current_position = start
        self.action_space = [0, 1, 2, 3]  # 上、下、左、右
        self.history = []  # 用于存储历史路径

    def step(self, action):
        x, y = self.current_position
        if action == 0:  # 上
            x -= 1
        elif action == 1:  # 下
            x += 1
        elif action == 2:  # 左
            y -= 1
        elif action == 3:  # 右
            y += 1

        if 0 <= x < self.maze.shape[0] and 0 <= y < self.maze.shape[1] and self.maze[x, y] == 0:
            # print(f"Moving to ({x}, {y})")
            self.current_position = (x, y)
            self.history.append(self.current_position)  # 记录历史位置

        reward = -1
        done = False

        if self.current_position == self.goal:
            reward = 100
            done = True

        return self.current_position, reward, done

    def reset(self):
        self.current_position = self.start
        self.history = [self.start]  # 重置历史路径
        return self.current_position

    def render(self, episode=None, if_save=False):
        # 创建一个图形
        fig, ax = plt.subplots(figsize=(20, 30))
        cmap = ListedColormap(['black', 'white'])  # 颜色映射：黑色为路径，白色为墙壁
        ax.imshow(self.maze, cmap=cmap)

        # 标记当前路径（包括历史路径）
        path_x, path_y = zip(*self.history)  # 拆分历史路径
        ax.plot(path_y, path_x, color="red", marker="o", markersize=5, label="Path")

        # 标记当前智能体位置
        cx, cy = self.current_position
        ax.plot(cy, cx, color="blue", marker="o", markersize=10, label="Current Position")  # 当前智能体位置

        # 标记目标点
        gx, gy = self.goal
        ax.plot(gy, gx, color="green", marker="o", markersize=10, label="Goal")  # 目标位置

        ax.set_title("Path Visualization")
        ax.set_xticks(np.arange(len(self.maze[0])))
        ax.set_yticks(np.arange(len(self.maze)))
        ax.grid(True)
        ax.legend()  # 添加图例

        if if_save and episode is not None:
            plt.savefig(f'IntroductiontoMachineLearningandNeuralNetworks\Problem8_Maze_RL\QlearningResult\maze_path_solution_episode_{episode}.png', dpi=300, bbox_inches='tight')
        
        # plt.show()
        # plt.close()

# Q-Learning Implementation
def q_learning(env, episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.1, render_interval=100):
    q_table = np.zeros((env.maze.shape[0], env.maze.shape[1], len(env.action_space)))
    
    for episode in tqdm(range(episodes), file=sys.stdout, desc="Training"):
        state = env.reset()
        done = False

        while not done:
            x, y = state
            if random.uniform(0, 1) < epsilon:
                action = random.choice(env.action_space)  # Explore
            else:
                action = np.argmax(q_table[x, y])  # Exploit

            next_state, reward, done = env.step(action)
            nx, ny = next_state

            old_value = q_table[x, y, action]
            next_max = np.max(q_table[nx, ny])

            # Q-value update
            q_table[x, y, action] = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)

            state = next_state

        # Render the environment at intervals
        if episode % render_interval == 0:
            env.render(episode, if_save=True)

    return q_table


# Generate path based on Q-Table
def generate_path(env, q_table):
    path = []
    state = env.reset()
    path.append(state)

    for _ in range(100):  # Limit steps to avoid infinite loops
        x, y = state
        action = np.argmax(q_table[x, y])
        next_state, _, done = env.step(action)

        path.append(next_state)
        state = next_state

        if done:
            break

    return path

def load_maze_from_txt(input_path):
    """从TXT文件中读取迷宫数组"""
    with open(input_path, 'r') as f:
        maze_array = [list(map(int, line.split())) for line in f]
    return maze_array


# Example usage
if __name__ == "__main__":
    # Define maze (0 = free, 1 = wall)
    maze = load_maze_from_txt('IntroductiontoMachineLearningandNeuralNetworks\Problem8_Maze_RL\mazeArray.txt')
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
