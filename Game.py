import math
import copy
import pyautogui
import numpy as np
from get_game import get_game


class Game:
    def __init__(self, game, gui=False):
        self.H, self.W = 10, 17
        self.is_gui = gui
        self.game = game
        self.score = 0
        self.nums = np.ndarray(game.shape)
        self.total_cells = self.H * self.W
        for i in range(self.H):
            for j in range(self.W):
                self.nums[i][j] = self.game[i][j].num

    def update(self, i, y, j, x):
        for idx in range(y + 1):
            for jdx in range(x + 1):
                self.game[i + idx][j + jdx].num = self.nums[i + idx][j + jdx]

    def print_game(self):
        print(f"Score: {self.score}")
        print()
        for i in range(self.H):
            for j in range(self.W):
                if self.game[i][j].num == 0:
                    print(" ", end=" ")
                else:
                    print(f"{int(self.game[i][j].num)}", end=" ")
            print()
        print()

    def drag(self, sx, sy, ex, ey):
        pyautogui.moveTo(self.game[sx][sy].x - 20, self.game[sx][sy].y - 20)
        pyautogui.dragTo(
            self.game[ex][ey].endx + 20,
            self.game[ex][ey].endy + 20,
            math.sqrt(((ex - sx) ** 2) + ((ey - sy) ** 2)) * 0.5,
            button="left",
        )

    def _compute_prefix_sums(self, nums: np.ndarray) -> np.ndarray:
        return np.cumsum(nums, axis=0)

    def _rectangle_sum(self, prefix: np.ndarray, top: int, bottom: int) -> np.ndarray:
        if top == 0:
            return prefix[bottom]
        return prefix[bottom] - prefix[top - 1]

    def _find_ten_rectangles(self, nums: np.ndarray):
        prefix = self._compute_prefix_sums(nums)
        for top in range(self.H):
            for bottom in range(top, self.H):
                col_sums = self._rectangle_sum(prefix, top, bottom)
                left = 0
                current_sum = 0
                for right, value in enumerate(col_sums):
                    current_sum += value
                    while current_sum > 10 and left <= right:
                        current_sum -= col_sums[left]
                        left += 1
                    while current_sum == 10 and left <= right:
                        cleared_cells = int(
                            np.count_nonzero(nums[top : bottom + 1, left : right + 1])
                        )
                        if cleared_cells > 0:
                            yield (top, bottom, left, right, cleared_cells)
                        current_sum -= col_sums[left]
                        left += 1

    def _hash_board(self, nums: np.ndarray) -> bytes:
        return nums.tobytes()

    def _search(self, nums: np.ndarray, depth: int, memo: dict, branch_limit: int = 4):
        key = (self._hash_board(nums), depth)
        if key in memo:
            return memo[key]

        cleared_so_far = self.total_cells - int(np.count_nonzero(nums))
        moves = sorted(self._find_ten_rectangles(nums), key=lambda m: m[4])
        if depth == 0 or not moves:
            memo[key] = (cleared_so_far, None)
            return memo[key]

        best_score = cleared_so_far
        best_move = None
        for top, bottom, left, right, _ in moves[:branch_limit]:
            child = np.array(nums, copy=True)
            child[top : bottom + 1, left : right + 1] = 0
            child_score, _ = self._search(child, depth - 1, memo, branch_limit)
            if child_score > best_score:
                best_score = child_score
                best_move = (top, bottom, left, right)
        memo[key] = (best_score, best_move)
        return memo[key]

    def _apply_move(self, move):
        top, bottom, left, right = move
        cleared_cells = int(np.count_nonzero(self.nums[top : bottom + 1, left : right + 1]))
        self.score += cleared_cells
        self.nums[top : bottom + 1, left : right + 1] = 0
        self.update(top, bottom - top, left, right - left)
        if self.is_gui:
            self.drag(sx=top, sy=left, ex=bottom, ey=right)
        return cleared_cells

    def _basic_solver(self):
        while True:
            moves = list(self._find_ten_rectangles(self.nums))
            if not moves:
                break
            top, bottom, left, right, _ = moves[0]
            self._apply_move((top, bottom, left, right))
        return self.score

    def _greedy_solver(self):
        while True:
            moves = list(self._find_ten_rectangles(self.nums))
            if not moves:
                break
            best_move = max(moves, key=lambda m: m[4])
            top, bottom, left, right, _ = best_move
            self._apply_move((top, bottom, left, right))
        return self.score

    def _dfs_solver(self, depth: int = 4, branch_limit: int = 4):
        memo = {}
        while True:
            _, best_move = self._search(self.nums, depth, memo, branch_limit)
            if best_move is None:
                break
            self._apply_move(best_move)
        return self.score

    def solve(self, solver="dfs", depth: int = 4, branch_limit: int = 4):
        if solver == "basic":
            return self._basic_solver()
        if solver in ("algorithm1", "greedy"):
            return self._greedy_solver()
        if solver == "dfs":
            return self._dfs_solver(depth=depth, branch_limit=branch_limit)
        raise ValueError(f"Unknown solver type: {solver}")


def get_idxs(game):
    return [idx[:4] for idx in game._find_ten_rectangles(game.nums)]


def greedy(tens_idxs, game, depth=0):
    scores = []
    if len(tens_idxs) > 1:
        for idx in tens_idxs:
            copy_game = copy.deepcopy(game)
            i, l, j, k = idx
            copy_game._apply_move((i, l, j, k))
            depth += 1
            copy_game.solve("basic")
            scores.append(copy_game.score)
    else:
        print(scores)
    return scores.index(max(scores)), max(scores)


if __name__ == "__main__":
    game = Game(get_game(confidence=0.8), False)
    c_game = copy.deepcopy(game)
    c_game.solve("basic")
    print(f"Naive: {c_game.score}")

    best_score = 0
    idxs = get_idxs(game)
    while len(idxs) > 10:
        idxs = get_idxs(game)
        dx, s = greedy(idxs, game)
        i, l, j, k = idxs[dx]
        game._apply_move((i, l, j, k))
        if s == best_score:
            break
        best_score = s
    print(f"Greedy:{best_score}")
