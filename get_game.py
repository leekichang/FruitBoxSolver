import os
import sys
import pyautogui
import numpy as np
from Apple import Apple

H, W = 10, 17
N_APPLE = H*W

def check_overlap(check_arr, new_pos):
    loss = np.array(check_arr)-np.array(new_pos)
    if np.all(np.power(np.sum(np.power(loss, 2), axis=1), 0.5) > 10):
        check_arr.append(new_pos)
        return False
    else:
        return True
    
def get_apples(confidence):
    applelist = []
    check_arr = []
    for idx in range(1, 10):
        pos = pyautogui.locateAllOnScreen(f'./figure/{idx}.png', confidence=confidence)
        overlap = False
        for _, p in enumerate(pos):
            if len(check_arr) > 0:
                overlap = check_overlap(check_arr, [p.left, p.top])
            else:
                check_arr = [[p.left, p.top]]
                
            if not overlap:
                applelist.append(Apple(idx, p.left, p.top, p.width, p.height))
    return applelist

def sort_apples(applelist):
    applelist = np.array(sorted(applelist, key=lambda x:x.y))
    for i in range(H):
        applelist[i*W:(i+1)*W]=np.array(sorted(applelist[i*W:(i+1)*W], key=lambda x:x.x))
    return applelist
 
def get_game(confidence=0.8, save=False):
    applelist = get_apples(confidence)
    
    assert len(applelist) == N_APPLE, f"Not Enough Apple Found, {len(applelist)} found"
    
    game = sort_apples(applelist)
    game = game.reshape(H, W)
    if save:
        n_games = len(os.listdir('./games'))
        np.save(f'./games/{n_games}.npy', game)
    return game

if __name__ == '__main__':
    confidence = sys.argv[1] if len(sys.argv) > 1 else 0.8
    game = get_game(confidence)