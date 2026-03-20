#!/usr/bin/env python3
"""
Default Dance - Braille Terminal Animation
No dependencies required. Just run: python3 dance.py
"""

import base64
import curses
import json
import time
import zlib

_DATA = (
%%DATA%%
)

FRAMES = json.loads(zlib.decompress(base64.b64decode(_DATA)))


def play(stdscr: curses.window) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(20)
    frame_duration = 1.0 / 30

    for frame_lines in FRAMES:
        t0 = time.monotonic()
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()

        frame_w = len(frame_lines[0]) if frame_lines else 0
        x_off = max(0, (cols - frame_w) // 2)
        y_off = max(0, (rows - len(frame_lines)) // 2)

        for i, line in enumerate(frame_lines):
            y = y_off + i
            if y >= rows - 1:
                break
            try:
                stdscr.addstr(y, x_off, line[: cols - x_off - 1])
            except curses.error:
                pass

        stdscr.refresh()

        if stdscr.getch() == ord("q"):
            return

        elapsed = time.monotonic() - t0
        if elapsed < frame_duration:
            time.sleep(frame_duration - elapsed)


if __name__ == "__main__":
    curses.wrapper(play)
