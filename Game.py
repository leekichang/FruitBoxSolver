import math
import copy
import pyautogui
import numpy as np
from get_game import get_game


class Game:
    def __init__(self, game, gui=True):
        self.H, self.W = 10, 17
        self.is_gui = gui
        self.game = game
        self.score = 0
        self.nums = np.array([[cell.num for cell in row] for row in game], dtype=np.int16)
        self.total_cells = self.H * self.W

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
        start_x = (self.game[sx][sy].x + self.game[sx][sy].endx)//2
        start_y = (self.game[sx][sy].y + self.game[sx][sy].endy)//2
        end_x = (self.game[ex][ey].x + self.game[ex][ey].endx)//2
        end_y = (self.game[ex][ey].y + self.game[ex][ey].endy)//2
        # pyautogui.moveTo(self.game[sx][sy].x, self.game[sx][sy].y)
        pyautogui.moveTo(start_x, start_y)
        #pyautogui.dragTo(self.game[ex][ey].endx, self.game[ex][ey].endy,\
        #    math.sqrt(((ex-sx)**2) + ((ey-sy)**2))*0.5, button='left')
        # pyautogui.dragTo(end_x, end_y, math.sqrt(((ex-sx)**2) + ((ey-sy)**2)), button='left')
        pyautogui.dragTo(end_x, end_y, 0.5*math.sqrt(((ex-sx)**2) + ((ey-sy)**2))**0.5, button='left')
        print(f'Dragged from ({sx}, {sy}) to ({ex}, {ey})')

    def _compute_prefix_sums(self, nums: np.ndarray) -> np.ndarray:
        return np.cumsum(nums, axis=0, dtype=np.int32)

    def _rectangle_sum(self, prefix: np.ndarray, top: int, bottom: int) -> np.ndarray:
        if top == 0:
            return prefix[bottom]
        return prefix[bottom] - prefix[top - 1]

    def _find_ten_rectangles(self, nums: np.ndarray):
        prefix_values = self._compute_prefix_sums(nums)
        prefix_nonzero = self._compute_prefix_sums((nums != 0).astype(np.uint8))
        for top in range(self.H):
            for bottom in range(top, self.H):
                col_sums = self._rectangle_sum(prefix_values, top, bottom)
                col_nonzero = self._rectangle_sum(prefix_nonzero, top, bottom)
                left = 0
                current_sum = 0
                current_nonzero = 0
                for right in range(len(col_sums)):
                    current_sum += col_sums[right]
                    current_nonzero += col_nonzero[right]
                    while current_sum > 10 and left <= right:
                        current_sum -= col_sums[left]
                        current_nonzero -= col_nonzero[left]
                        left += 1
                    while current_sum == 10 and left <= right:
                        if current_nonzero > 0:
                            yield (top, bottom, left, right, int(current_nonzero))
                        current_sum -= col_sums[left]
                        current_nonzero -= col_nonzero[left]
                        left += 1

    def _hash_board(self, nums: np.ndarray) -> bytes:
        return nums.tobytes()

    def _greedy_min_rollout(self, nums: np.ndarray) -> int:
        board = np.array(nums, copy=True)
        cleared = self.total_cells - int(np.count_nonzero(board))

        while True:
            moves = list(self._find_ten_rectangles(board))
            if not moves:
                break

            top, bottom, left, right, cleared_cells = min(moves, key=lambda m: m[4])
            cleared += cleared_cells
            board[top : bottom + 1, left : right + 1] = 0

        return cleared

    def _search(self, nums: np.ndarray, depth: int, memo: dict, branch_limit: int = 4):
        key = (self._hash_board(nums), depth)
        if key in memo:
            return memo[key]

        moves = sorted(self._find_ten_rectangles(nums), key=lambda m: m[4])
        if depth == 0 or not moves:
            memo[key] = (self._greedy_min_rollout(nums), None)
            return memo[key]

        best_score = -1
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

    def _heuristic_cleared(self, nums: np.ndarray) -> int:
        return self.total_cells - int(np.count_nonzero(nums))

    def _beam_step(self, nums: np.ndarray, depth: int, beam_width: int):
        frontier = [(self._heuristic_cleared(nums), nums.copy(), [])]
        best_score = frontier[0][0]
        best_path = []

        for _ in range(depth):
            next_frontier = []
            for cleared, board, path in frontier:
                for top, bottom, left, right, cleared_cells in self._find_ten_rectangles(board):
                    child = np.array(board, copy=True)
                    child[top : bottom + 1, left : right + 1] = 0
                    child_cleared = self._heuristic_cleared(child)
                    new_path = path + [(top, bottom, left, right)]
                    if child_cleared > best_score or (
                        child_cleared == best_score and len(new_path) > len(best_path)
                    ):
                        best_score = child_cleared
                        best_path = new_path
                    next_frontier.append((child_cleared, child, new_path, cleared_cells))
            if not next_frontier:
                break
            next_frontier.sort(key=lambda x: (x[0], x[3]), reverse=True)
            frontier = [(c, b, p) for c, b, p, _ in next_frontier[:beam_width]]

        return best_path

    def _beam_solver(self, depth: int = 6, beam_width: int = 24):
        while True:
            moves = list(self._find_ten_rectangles(self.nums))
            if not moves:
                break

            best_path = self._beam_step(self.nums, depth, beam_width)
            if not best_path:
                top, bottom, left, right, _ = max(moves, key=lambda m: m[4])
                self._apply_move((top, bottom, left, right))
            else:
                for move in best_path:
                    self._apply_move(move)
        return self.score

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
        if solver == "beam":
            return self._beam_solver(depth=depth, beam_width=branch_limit)
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
    game = Game(get_game(confidence=0.975), True)
    c_game = copy.deepcopy(game)
    c_game.is_gui = False
    c_game.solve("dfs", depth=5, branch_limit=4)
    c_game.print_game()
    game.solve("dfs", depth=5, branch_limit=4)

    # best_score = 0
    # idxs = get_idxs(game)
    # while len(idxs) > 10:
    #     idxs = get_idxs(game)
    #     dx, s = greedy(idxs, game)
    #     i, l, j, k = idxs[dx]
    #     game._apply_move((i, l, j, k))
    #     if s == best_score:
    #         break
    #     best_score = s
    # print(f"Greedy:{best_score}")
