import heapq
from typing import List, Tuple, Optional
import numpy as np

from src.environment import Environment

Coord = Tuple[int, int, int]


class AStarPlanner:
    """
    在 3D 网格环境中运行 A* 路径规划。
    """

    def __init__(self, env: Environment):
        self.env = env
        self.size = env.size

    def heuristic(self, a: Coord, b: Coord) -> float:
        """曼哈顿距离启发式"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2])

    def get_neighbors(self, node: Coord) -> List[Coord]:
        """6邻接网格（不斜走）"""
        x, y, z = node
        neighbors = []

        moves = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1)
        ]

        for dx, dy, dz in moves:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < self.size and 0 <= ny < self.size and 0 <= nz < self.size:
                candidate = (nx, ny, nz)
                if not self.env.is_blocked(candidate):
                    neighbors.append(candidate)


        return neighbors

    def reconstruct_path(self, came_from: dict, current: Coord) -> List[Coord]:
        """根据记录重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def plan(self) -> Optional[List[Coord]]:
        """执行 A* 搜索"""
        if self.env.start is None or self.env.goal is None:
            raise ValueError("Start and goal must be set.")

        start: Coord = tuple(int(v) for v in self.env.start)
        goal: Coord = tuple(int(v) for v in self.env.goal)

        open_set = []
        counter = 0
        heapq.heappush(open_set, (0.0, counter, start))

        came_from: dict[Coord, Coord] = {}
        g_score: dict[Coord, float] = {start: 0.0}
        f_score: dict[Coord, float] = {start: self.heuristic(start, goal)}
        open_set_hash = {start}

        while open_set:
            _, _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor in self.get_neighbors(current):
                tentative_g = g_score[current] + 1.0

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.heuristic(neighbor, goal)

                    if neighbor not in open_set_hash:
                        counter += 1
                        heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))
                        open_set_hash.add(neighbor)

        return None  # 无路径
