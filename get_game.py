import os
import sys
import pyautogui
import numpy as np
from PIL import Image
from Apple import Apple

H, W = 10, 17
N_APPLE = H * W

def check_overlap(check_arr, new_pos, threshold):
    loss = np.array(check_arr) - np.array(new_pos)
    if np.all(np.power(np.sum(np.power(loss, 2), axis=1), 0.5) > threshold):
        check_arr.append(new_pos)
        return False
    return True


def locate_game_area(anchor_confidence=0.8, padding=30):
    anchors = []
    for anchor in ("playButton.png", "resetButton.png"):
        try:
            pos = pyautogui.locateOnScreen(f"./figure/{anchor}", confidence=anchor_confidence)
            if pos:
                anchors.append(pos)
        except OSError:
            # In headless environments image loading can fail; fall back later.
            continue

    if not anchors:
        return None, None

    left = min(a.left for a in anchors)
    top = min(a.top for a in anchors)
    right = max(a.left + a.width for a in anchors)
    bottom = max(a.top + a.height for a in anchors)

    left = max(0, left - padding)
    top = max(0, top - padding)
    width = (right - left) + (padding * 2)
    height = (bottom - top) + (padding * 2)

    tile_width = width / W
    tile_height = height / H
    estimated_tile_size = int(round(min(tile_width, tile_height)))
    return (left, top, width, height), estimated_tile_size


def resize_template(path, target_size):
    image = Image.open(path)
    return image.resize((int(target_size), int(target_size)))


def get_apples(confidence, region=None, tile_size=None, scale_grid=None):
    applelist = []
    check_arr = []
    overlap_threshold = 0.6 * tile_size if tile_size else 10
    scales = scale_grid if scale_grid else [1.0]

    for idx in range(1, 10):
        for scale in scales:
            target_size = tile_size * scale if tile_size else None
            image_path = f"./figure/{idx}.png"
            template = resize_template(image_path, target_size) if target_size else image_path
            pos = pyautogui.locateAllOnScreen(template, confidence=confidence, region=region)
            overlap = False
            for _, p in enumerate(pos):
                if len(check_arr) > 0:
                    overlap = check_overlap(check_arr, [p.left, p.top], overlap_threshold)
                else:
                    check_arr = [[p.left, p.top]]

                if not overlap:
                    applelist.append(Apple(idx, p.left, p.top, p.width, p.height))
    return applelist

def sort_apples(applelist):
    applelist = np.array(sorted(applelist, key=lambda x: x.y))
    for i in range(H):
        applelist[i * W:(i + 1) * W] = np.array(
            sorted(applelist[i * W:(i + 1) * W], key=lambda x: x.x)
        )
    return applelist

def get_game(confidence=0.8, save=False):
    region, tile_size = locate_game_area()
    initial_scale_grid = [1.0, 0.95, 1.05] if tile_size else [1.0]
    applelist = get_apples(confidence, region=region, tile_size=tile_size, scale_grid=initial_scale_grid)

    if len(applelist) < N_APPLE:
        retry_scales = [0.85, 0.925, 1.0, 1.075, 1.15]
        applelist = get_apples(confidence, region=region, tile_size=tile_size, scale_grid=retry_scales)

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
    print(game)