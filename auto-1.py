#!/usr/bin/env python3
"""
keep_awake.py
Keep the screen awake for 8 hours by gently moving the mouse using autopy.
If autopy isn't available, the script will attempt to use pyautogui as a fallback.

Usage:
    python keep_awake.py
"""

import time
import random
import sys

# Total duration (seconds): 8 hours = 8 * 3600 = 28800 seconds
TOTAL_SECONDS = 8 * 3600  # 28800
# How often to move the mouse (seconds). Adjust if you want more/less frequent nudges.
INTERVAL_SECONDS = 60  # move every 60 seconds
# Maximum pixel displacement for each nudge (keeps movement tiny)
MAX_DELTA = 6

try:
    import autopy
    backend = "autopy"
except Exception:
    try:
        import pyautogui
        backend = "pyautogui"
    except Exception as e:
        print("Error: Neither 'autopy' nor 'pyautogui' is installed.")
        print("Install one with: pip install autopy    OR    pip install pyautogui")
        sys.exit(1)

def move_mouse_autopy():
    x, y = autopy.mouse.get_pos()
    dx = random.randint(-MAX_DELTA, MAX_DELTA)
    dy = random.randint(-MAX_DELTA, MAX_DELTA)
    new_x = max(0, x + dx)
    new_y = max(0, y + dy)
    print(f"Moving mouse to ({new_x}, {new_y})")

    # autopy expects integers
    autopy.mouse.move(int(new_x), int(new_y))

def move_mouse_pyautogui():
    x, y = pyautogui.position()
    dx = random.randint(-MAX_DELTA, MAX_DELTA)
    dy = random.randint(-MAX_DELTA, MAX_DELTA)
    new_x = max(0, x + dx)
    new_y = max(0, y + dy)
    print(f"Moving mouse to ({new_x}, {new_y})")
    pyautogui.moveTo(new_x, new_y, duration=0.15)

def main():
    start = time.time()
    end_time = start + TOTAL_SECONDS
    print(f"Keeping screen awake for {TOTAL_SECONDS} seconds (~8 hours). Press Ctrl+C to stop.")
    try:
        while time.time() < end_time:
            if backend == "autopy":
                move_mouse_autopy()
            else:
                move_mouse_pyautogui()
            # sleep until next interval
            time_remaining = end_time - time.time()
            # Sleep smallest of INTERVAL_SECONDS or remaining time to finish accurately
            time.sleep(min(INTERVAL_SECONDS, max(0, time_remaining)))
    except KeyboardInterrupt:
        print("\nStopped by user.")
    print("Done — 8 hours elapsed (or script stopped).")

if __name__ == "__main__":
    main()
