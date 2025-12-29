import numpy as np
from typing import Set, Tuple

Coord = Tuple[int, int, int]


class Environment:
    def __init__(self, size: int = 50, obstacle_ratio: float = 0.05, seed: int | None = None):
        """
        :param size: 立方体边长，坐标范围 [0, size)
        :param obstacle_ratio: 障碍占比（0.05 = 5%格子是障碍）
        :param seed: 随机种子（可复现实验）
        """
        self.size = size
        self.obstacle_ratio = obstacle_ratio
        self.rng = np.random.default_rng(seed)
        self.static_obstacles: set[Coord] = set()

        self.start = None
        self.goal = None
        self.obstacles: Set[Coord] = set()

    def random_point(self) -> np.ndarray:
        return self.rng.integers(0, self.size, size=3)

    def generate_start_goal(self):
        self.start = self.random_point()
        self.goal = self.random_point()
        while np.array_equal(self.start, self.goal):
            self.goal = self.random_point()

    def generate_obstacles(self):
        """
        随机生成障碍，避免生成到 start / goal 上。
        """
        if self.start is None or self.goal is None:
            raise ValueError("Call generate_start_goal() before generate_obstacles().")

        total_cells = self.size ** 3
        num_obs = int(total_cells * self.obstacle_ratio)

        start_t = tuple(int(v) for v in self.start)
        goal_t = tuple(int(v) for v in self.goal)

        self.obstacles.clear()

        while len(self.obstacles) < num_obs:
            p = self.random_point()
            pt = tuple(int(v) for v in p)
            if pt != start_t and pt != goal_t:
                self.obstacles.add(pt)

    def is_blocked(self, node: Coord) -> bool:
        return node in self.obstacles

    def __repr__(self) -> str:
        return f"Environment(size={self.size}, start={self.start}, goal={self.goal}, obstacles={len(self.obstacles)})"
    def move_obstacles(self, move_prob: float = 0.3):
        """
        让一部分障碍随机移动一格（模拟动态障碍）
        """
        new_obstacles = set()

        moves = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
            (0, 0, 0),  # 不动
        ]

        for (x, y, z) in self.obstacles:
            if self.rng.random() < move_prob:
                dx, dy, dz = moves[self.rng.integers(0, len(moves))]
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                    new_obstacles.add((nx, ny, nz))
                else:
                    new_obstacles.add((x, y, z))
            else:
                new_obstacles.add((x, y, z))

        self.obstacles = new_obstacles
    def generate_building_block(
        self,
        base_x: int,
        base_y: int,
        width: int,
        depth: int,
        height: int,
    ):
        """
        生成一个长方体建筑（高楼）
        """
        for x in range(base_x, base_x + width):
            for y in range(base_y, base_y + depth):
                for z in range(0, min(height, self.size)):
                    if 0 <= x < self.size and 0 <= y < self.size:
                        self.static_obstacles.add((x, y, z))
    def is_blocked(self, node: Coord) -> bool:
        if node in self.static_obstacles:
            return True
        if hasattr(self, "dynamic_obstacles") and node in set(self.dynamic_obstacles):
            return True
        return False
