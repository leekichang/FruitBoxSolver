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
        for i in range(self.H):
            for j in range(self.W):
                self.nums[i][j]=self.game[i][j].num
        
    def update(self, i, y, j, x):
        for idx in range(y+1):
            for jdx in range(x+1):
                self.game[i+idx][j+jdx].num = self.nums[i+idx][j+jdx]
                
    def print_game(self):
        print(f'Score: {self.score}')
        print()
        for i in range(self.H):
            for j in range(self.W):
                if self.game[i][j].num == 0:
                    print(' ', end=' ')
                else:
                    print(f'{int(self.game[i][j].num)}', end=' ')
            print()
        print()
    
    def drag(self, sx, sy, ex, ey):
        pyautogui.moveTo(self.game[sx][sy].x-20, self.game[sx][sy].y-20)
        pyautogui.dragTo(self.game[ex][ey].endx+20, self.game[ex][ey].endy+20,\
            math.sqrt(((ex-sx)**2) + ((ey-sy)**2))*0.5, button='left')
    
    def solve(self, solver='basic'):
        #self.print_game()
        if solver=='basic':     # naive approach
            new = True
            while new == True:
                new = False
                for i in range(self.H):
                    for j in range(self.W):
                        sum = 0
                        x, y = 0, 0
                        while sum <= 10 and i+y+1 < self.H+1:
                            while sum <= 10 and j+x+1 < self.W+1:
                                if self.nums[i, j] == 0:
                                    pass
                                else:
                                    sum = np.sum(self.nums[i:i+y+1, j:j+x+1])
                                    if sum == 10:
                                        if self.is_gui:
                                            self.drag(sx=i, sy=j, ex=i+y, ey=j+x)
                                        self.score += len(np.nonzero(self.nums[i:i+y+1, j:j+x+1].reshape(-1))[0])
                                        self.nums[i:i+y+1, j:j+x+1] = np.zeros(self.nums[i:i+y+1, j:j+x+1].shape)
                                        self.update(i, y, j, x)
                                        #self.print_game()
                                        new = True
                                x += 1
                            y += 1
                            x = 0
                            sum = np.sum(self.nums[i:i+y+1, j:j+x+1])
        elif solver=='algorithm1':
            new = True
            while new == True:
                new = False
                tens_idxs = []
                for i in range(self.H):
                    for j in range(self.W):
                        sum = 0
                        x, y = 0, 0
                        while sum <= 10 and i+y+1 < self.H+1:
                            while sum <= 10 and j+x+1 < self.W+1:
                                if self.nums[i, j] == 0:
                                    pass
                                else:
                                    sum = np.sum(self.nums[i:i+y+1, j:j+x+1])
                                    if sum == 10:
                                        tens_idxs.append([i, i+y+1, j, j+x+1])      
                                        new = True
                                x += 1
                            y += 1
                            x = 0
                            sum = np.sum(self.nums[i:i+y+1, j:j+x+1])
                if new == True:
                    best_box_score = -1
                    best_box_idx = []
                    for idx in tens_idxs:
                        i, l, j, k = idx
                        s = len(np.nonzero(self.nums[i:l, j:k].reshape(-1))[0])
                        if s > best_box_score:
                            best_box_score = s
                            best_box_idx = idx
                    i, l, j, k = best_box_idx
                    if self.is_gui:
                        self.drag(sx=i, sy=j, ex=l-1, ey=k-1)
                    self.score += best_box_score
                    self.nums[i:l, j:k] = np.zeros(self.nums[i:l, j:k].shape)
                    self.update(i, l-i-1, j, k-j-1)
                    #self.print_game()
def get_idxs(game):
    tens_idxs = []
    for i in range(game.H):
        for j in range(game.W):
            sum = 0
            x, y = 0, 0
            while sum <= 10 and i+y+1 < game.H+1:
                while sum <= 10 and j+x+1 < game.W+1:
                    if game.nums[i, j] == 0:
                        pass
                    else:
                        sum = np.sum(game.nums[i:i+y+1, j:j+x+1])
                        if sum == 10:
                            tens_idxs.append([i, i+y+1, j, j+x+1])
                    x += 1
                y += 1
                x = 0
                sum = np.sum(game.nums[i:i+y+1, j:j+x+1])
    return tens_idxs

def greedy(tens_idxs, game, depth=0):
    scores = []
    if len(tens_idxs) > 1:
    #    print(f"len={len(tens_idxs)}")
        for jdx, idx in enumerate(tens_idxs):
            copy_game = copy.deepcopy(game)
            i, l, j, k = idx
            copy_game.score += len(np.nonzero(copy_game.nums[i:l, j:k].reshape(-1))[0])
            copy_game.nums[i:l, j:k] = np.zeros(copy_game.nums[i:l, j:k].shape)
            copy_game.update(i, l-i-1, j, k-j-1)
            depth += 1
            copy_game.solve('basic')
            scores.append(copy_game.score)
            #scores.append(copy_game.score)
    else:
        #game.solve(solver='basic')
        print(scores)
    #print(max(scores), scores.index(max(scores)), len(scores))
    return scores.index(max(scores)), max(scores)
       
if __name__ == '__main__':
    game = Game(get_game(confidence=0.8), False)
    c_game = copy.deepcopy(game)
    c_game.solve()
    print(f'Naive: {c_game.score}')
    
    best_score = 0
    idxs = get_idxs(game)
    while len(idxs) > 10:
        idxs = get_idxs(game)
        dx, s = greedy(idxs, game)
        i,l,j,k = idxs[dx]
        game.score += len(np.nonzero(game.nums[i:l, j:k].reshape(-1))[0])
        game.nums[i:l, j:k] = np.zeros(game.nums[i:l, j:k].shape)
        game.update(i, l-i-1, j, k-j-1)
        if s == best_score:
            break
        best_score = s
    print(f'Greedy:{best_score}')
    #import copy
    #game1 = copy.deepcopy(game)
    #game.solve(solver='basic')
    #game1.solve(solver='algorithm1')
    #print(game.score, game1.score)