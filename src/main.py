import numpy as np
from src.environment import Environment
  # 如果报错，改成 from .environment import Environment 试试
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 只是为了激活 3D 支持，pycharm/VS Code 可能提示未使用
from src.planner import AStarPlanner
from matplotlib.animation import FuncAnimation


def plot_environment_with_path(env: Environment, path):
    """画 3D 空间 + 起点/终点"""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 画起点和终点
    start = env.start
    goal = env.goal

    ax.scatter(start[0], start[1], start[2], marker='o', s=50, label="Start")
    ax.scatter(goal[0], goal[1], goal[2], marker='^', s=50, label="Goal")

    # 画障碍物（点云）
    if len(env.obstacles) > 0:
        ox = [p[0] for p in env.obstacles]
        oy = [p[1] for p in env.obstacles]
        oz = [p[2] for p in env.obstacles]
        ax.scatter(ox, oy, oz, s=20, alpha=0.8,  c="black",label="Obstacles")

    if path is not None and len(path) > 0:
        xs = [p[0] for p in path]
        ys = [p[1] for p in path]
        zs = [p[2] for p in path]
        ax.plot(xs, ys, zs, linewidth=2, label="Path")

    # ===== 静态基础设施障碍（红色，长期存在）=====
    if len(env.static_obstacles) > 0:
        sx = [p[0] for p in env.static_obstacles]
        sy = [p[1] for p in env.static_obstacles]
        sz = [p[2] for p in env.static_obstacles]

        ax.scatter(
            sx, sy, sz,
            c="red",
            s=10,
            alpha=0.8,
            label="Static Infrastructure"
        )

    # 设置坐标轴范围
    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_zlim(0, env.size)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D UAV Environment (Start & Goal)")

    ax.legend()
    plt.show()

def simulate_with_replanning(env: Environment, planner: AStarPlanner, max_steps: int = 200):
    current = tuple(int(v) for v in env.start)
    goal = tuple(int(v) for v in env.goal)

    uav_history = [current]
    dyn_history = [list(getattr(env, "dynamic_obstacles", []))]  # 记录 t=0 的动态障碍位置
    replans = 0

    step = 0
    while current != goal and step < max_steps:
        path = planner.plan()
        if path is None or len(path) < 2:
            print("No path available, stopping.")
            break

        next_step = path[1]

        # 先更新动态障碍（你现在用 move_obstacles 或 step_dynamic_obstacles 都行）
        # 这里假设你有 env.step_dynamic_obstacles()
        if hasattr(env, "step_dynamic_obstacles"):
            env.step_dynamic_obstacles()
        else:
            env.move_obstacles(move_prob=0.3)

        # 记录动态障碍位置（更新后的位置）
        dyn_history.append(list(getattr(env, "dynamic_obstacles", [])))

        # 如果下一步被挡住，触发重规划（不走）
        if env.is_blocked(next_step):
            replans += 1
            # UAV 原地不动，也要记录一帧（保持 uav_history 与 dyn_history 对齐）
            uav_history.append(current)
            step += 1
            continue

        # 安全，走一步
        current = next_step
        env.start = np.array(current)
        uav_history.append(current)

        step += 1

    return uav_history, dyn_history, replans
def animate_simulation(env: Environment, uav_hist, dyn_hist):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_zlim(0, env.size)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title("3D UAV Simulation (Animated)")

    # --- 静态：起点/终点 ---
    start = uav_hist[0]
    goal = tuple(int(v) for v in env.goal)
    ax.scatter(start[0], start[1], start[2], marker='o', s=60, label="Start")
    ax.scatter(goal[0], goal[1], goal[2], marker='^', s=60, label="Goal")

    # --- 静态：红色建筑（长期存在）---
    if hasattr(env, "static_obstacles") and len(env.static_obstacles) > 0:
        sx = [p[0] for p in env.static_obstacles]
        sy = [p[1] for p in env.static_obstacles]
        sz = [p[2] for p in env.static_obstacles]
        ax.scatter(sx, sy, sz, c="red", s=10, alpha=0.7, label="Static Infrastructure")

    # --- 初始动态障碍（会更新）---
    dyn0 = dyn_hist[0] if len(dyn_hist) > 0 else []
    dx = [p[0] for p in dyn0]; dy = [p[1] for p in dyn0]; dz = [p[2] for p in dyn0]
    dyn_scatter = ax.scatter(dx, dy, dz, s=25, alpha=0.8, label="Dynamic Obstacles")

    # --- UAV 点（会更新）---
    u0 = uav_hist[0]
    uav_scatter = ax.scatter([u0[0]], [u0[1]], [u0[2]], s=80, label="UAV")

    # --- 轨迹线（逐步增长）---
    traj_line, = ax.plot([u0[0]], [u0[1]], [u0[2]], linewidth=2, label="Trajectory")

    ax.legend()

    def update(frame):
        # UAV 位置
        ux, uy, uz = uav_hist[frame]
        uav_scatter._offsets3d = ([ux], [uy], [uz])

        # 动态障碍位置
        if frame < len(dyn_hist):
            dd = dyn_hist[frame]
        else:
            dd = dyn_hist[-1] if len(dyn_hist) > 0 else []

        dx = [p[0] for p in dd]; dy = [p[1] for p in dd]; dz = [p[2] for p in dd]
        dyn_scatter._offsets3d = (dx, dy, dz)

        # 轨迹线更新
        xs = [p[0] for p in uav_hist[:frame+1]]
        ys = [p[1] for p in uav_hist[:frame+1]]
        zs = [p[2] for p in uav_hist[:frame+1]]
        traj_line.set_data(xs, ys)
        traj_line.set_3d_properties(zs)

        return uav_scatter, dyn_scatter, traj_line

    frames = min(len(uav_hist), len(dyn_hist))
    ani = FuncAnimation(fig, update, frames=frames, interval=80, blit=False)
    plt.show()


def main(env: Environment):
    planner = AStarPlanner(env)
    path = planner.plan()

    if path is None:
        print("No path found.")
    else:
        print(f"Path length: {len(path)} steps")

    plot_environment_with_path(env, path)

if __name__ == "__main__":
    env = Environment(size=30, obstacle_ratio=0.01, seed=42)
    env.generate_start_goal()
    env.generate_obstacles()

    # 建筑（红色）
    env.generate_building_block(base_x=8, base_y=8, width=5, depth=5, height=20)
    env.generate_building_block(base_x=18, base_y=4, width=3, depth=3, height=15)

    # 动态障碍（如果你实现了这个方法）
    if hasattr(env, "generate_dynamic_obstacles"):
        env.generate_dynamic_obstacles(num_dynamic=25, seed=7)

    planner = AStarPlanner(env)
    uav_hist, dyn_hist, replans = simulate_with_replanning(env, planner, max_steps=200)

    print("frames:", len(uav_hist))
    print("replans:", replans)

    animate_simulation(env, uav_hist, dyn_hist)


