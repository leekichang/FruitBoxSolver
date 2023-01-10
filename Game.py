from get_game import get_game
import numpy as np

class Game:
    def __init__(self, game):
        self.H, self.W = 10, 17
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
        
    def solve(self):
        self.print_game()
        new = True
        while new == True:
            new = False
            for i in range(self.H):
                for j in range(self.W):
                    sum = 0
                    x, y = 0, 0
                    while sum <= 10 and i+y+1 < self.H+1:
                        while sum <= 10 and j+x+1 < self.W+1:
                            sum = np.sum(self.nums[i:i+y+1, j:j+x+1])
                            if sum == 10:                                
                                self.score += len(np.nonzero(self.nums[i:i+y+1, j:j+x+1].reshape(-1))[0])
                                self.nums[i:i+y+1, j:j+x+1] = np.zeros(self.nums[i:i+y+1, j:j+x+1].shape)
                                self.update(i, y, j, x)
                                self.print_game()
                                new = True
                            x += 1
                        y += 1
                        x = 0
                        sum = np.sum(self.nums[i:i+y+1, j:j+x+1])
                
                    
                    
if __name__ == '__main__':
    game = Game(get_game(confidence=0.8))
    game.solve()
    