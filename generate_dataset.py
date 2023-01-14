import os
import sys
import time
import pyautogui
import numpy as np
from tqdm import tqdm
from get_game import get_game

if __name__ == '__main__':
    confidence = sys.argv[1] if len(sys.argv) > 1 else 0.99
    #for i in range(100):
    for i in tqdm(range(863)):
        pos = pyautogui.locateAllOnScreen(f'./figure/playButton.png', confidence=confidence)
        for p in pos:
            pyautogui.click(x=p.left+(p.width/2), y=p.top+(p.height/2), button='left')
            break
        
        time.sleep(1)
        
        game = get_game(confidence=0.8, save=True)
        
        pos = pyautogui.locateAllOnScreen(f'./figure/resetButton.png', confidence=confidence)
        for p in pos:
            pyautogui.click(x=p.left+(p.width/2), y=p.top+(p.height/2), button='left')
            break 
        
