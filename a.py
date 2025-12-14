import random
import sys
import tkinter as tk
from typing import Dict, List, Optional, Tuple


Coord = Tuple[int, int]
ShapeRotations = List[List[Coord]]


class TetrisGame:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Tetris")
        self.cell = 30
        self.cols = 10
        self.rows = 20
        self.width = self.cols * self.cell
        self.height = self.rows * self.cell

        self.canvas = tk.Canvas(
            root,
            width=self.width,
            height=self.height,
            bg="#111827",
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)

        self.side = tk.Frame(root, bg="#0b1220")
        self.side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.info = tk.Label(
            self.side,
            text="",
            justify=tk.LEFT,
            bg="#0b1220",
            fg="#e5e7eb",
            font=("Consolas", 12),
        )
        self.info.pack(anchor="nw", padx=10, pady=10)

        self.btn_new = tk.Button(
            self.side,
            text="New Game",
            command=self.reset,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
        )
        self.btn_new.pack(anchor="nw", padx=10, pady=(0, 10))

        self.root.bind("<Left>", lambda _: self.move(-1, 0))
        self.root.bind("<Right>", lambda _: self.move(1, 0))
        self.root.bind("<Down>", lambda _: self.soft_drop())
        self.root.bind("<Up>", lambda _: self.rotate())
        self.root.bind("<space>", lambda _: self.hard_drop())
        self.root.bind("p", lambda _: self.toggle_pause())
        self.root.bind("P", lambda _: self.toggle_pause())
        self.root.bind("n", lambda _: self.reset())
        self.root.bind("N", lambda _: self.reset())

        self.shapes: Dict[str, ShapeRotations] = {
            "I": [
                [(0, 0), (1, 0), (2, 0), (3, 0)],
                [(2, -1), (2, 0), (2, 1), (2, 2)],
            ],
            "O": [
                [(0, 0), (1, 0), (0, 1), (1, 1)],
            ],
            "T": [
                [(1, 0), (0, 1), (1, 1), (2, 1)],
                [(1, 0), (1, 1), (2, 1), (1, 2)],
                [(0, 1), (1, 1), (2, 1), (1, 2)],
                [(1, 0), (0, 1), (1, 1), (1, 2)],
            ],
            "S": [
                [(1, 0), (2, 0), (0, 1), (1, 1)],
                [(1, 0), (1, 1), (2, 1), (2, 2)],
            ],
            "Z": [
                [(0, 0), (1, 0), (1, 1), (2, 1)],
                [(2, 0), (1, 1), (2, 1), (1, 2)],
            ],
            "J": [
                [(0, 0), (0, 1), (1, 1), (2, 1)],
                [(1, 0), (2, 0), (1, 1), (1, 2)],
                [(0, 1), (1, 1), (2, 1), (2, 2)],
                [(1, 0), (1, 1), (0, 2), (1, 2)],
            ],
            "L": [
                [(2, 0), (0, 1), (1, 1), (2, 1)],
                [(1, 0), (1, 1), (1, 2), (2, 2)],
                [(0, 1), (1, 1), (2, 1), (0, 2)],
                [(0, 0), (1, 0), (1, 1), (1, 2)],
            ],
        }

        self.colors = {
            "I": "#06b6d4",
            "O": "#fbbf24",
            "T": "#a855f7",
            "S": "#22c55e",
            "Z": "#ef4444",
            "J": "#3b82f6",
            "L": "#f97316",
        }

        self.reset()

    def reset(self) -> None:
        self.board: List[List[Optional[str]]] = [
            [None for _ in range(self.cols)] for _ in range(self.rows)
        ]
        self.score = 0
        self.lines_cleared = 0
        self.level = 1
        self.tick_ms = 600
        self.paused = False
        self.game_over = False
        self.current = self.spawn_piece()
        self.next_piece = self.random_piece()
        self.schedule_tick()
        self.draw()

    def random_piece(self) -> Tuple[str, int, int, int]:
        shape = random.choice(list(self.shapes.keys()))
        return shape, 0, self.cols // 2 - 2, 0

    def spawn_piece(self) -> Tuple[str, int, int, int]:
        piece = getattr(self, "next_piece", None) or self.random_piece()
        self.next_piece = self.random_piece()
        return piece

    def schedule_tick(self) -> None:
        # after_cancel는 유효한 id가 필요하므로 먼저 존재 여부를 확인한다.
        tick_id = getattr(self, "_tick_after", None)
        if tick_id:
            self.root.after_cancel(tick_id)
        self._tick_after = self.root.after(self.tick_ms, self.tick)

    def tick(self) -> None:
        if self.paused or self.game_over:
            self.schedule_tick()
            return
        moved = self.move(0, 1)
        if not moved:
            self.lock_piece()
        self.schedule_tick()

    def rotate(self) -> None:
        shape, rot, x, y = self.current
        rotations = self.shapes[shape]
        next_rot = (rot + 1) % len(rotations)
        if not self.collides(shape, next_rot, x, y):
            self.current = (shape, next_rot, x, y)
            self.draw()

    def move(self, dx: int, dy: int) -> bool:
        shape, rot, x, y = self.current
        if not self.collides(shape, rot, x + dx, y + dy):
            self.current = (shape, rot, x + dx, y + dy)
            self.draw()
            return True
        return False

    def soft_drop(self) -> None:
        if not self.move(0, 1):
            self.lock_piece()

    def hard_drop(self) -> None:
        while self.move(0, 1):
            pass
        self.lock_piece()

    def collides(self, shape: str, rot: int, x: int, y: int) -> bool:
        for cx, cy in self.shapes[shape][rot]:
            bx, by = x + cx, y + cy
            if bx < 0 or bx >= self.cols or by >= self.rows:
                return True
            if by >= 0 and self.board[by][bx] is not None:
                return True
        return False

    def lock_piece(self) -> None:
        shape, rot, x, y = self.current
        for cx, cy in self.shapes[shape][rot]:
            bx, by = x + cx, y + cy
            if by < 0:
                self.game_over = True
                self.draw()
                return
            self.board[by][bx] = shape
        cleared = self.clear_lines()
        if cleared:
            self.score += (cleared ** 2) * 100
            self.lines_cleared += cleared
            self.level = 1 + self.lines_cleared // 10
            self.tick_ms = max(120, 600 - (self.level - 1) * 40)
        self.current = self.spawn_piece()
        if self.collides(*self.current):
            self.game_over = True
        self.draw()

    def clear_lines(self) -> int:
        new_board = [row for row in self.board if any(cell is None for cell in row)]
        cleared = self.rows - len(new_board)
        for _ in range(cleared):
            new_board.insert(0, [None for _ in range(self.cols)])
        self.board = new_board
        return cleared

    def toggle_pause(self) -> None:
        if self.game_over:
            return
        self.paused = not self.paused
        self.draw()

    def draw_block(self, x: int, y: int, color: str) -> None:
        x0 = x * self.cell
        y0 = y * self.cell
        x1 = x0 + self.cell
        y1 = y0 + self.cell
        self.canvas.create_rectangle(
            x0,
            y0,
            x1,
            y1,
            fill=color,
            outline="#0f172a",
            width=2,
        )

    def draw(self) -> None:
        self.canvas.delete("all")
        # Draw settled blocks
        for y in range(self.rows):
            for x in range(self.cols):
                if self.board[y][x]:
                    self.draw_block(x, y, self.colors[self.board[y][x]])  # type: ignore[arg-type]
        # Draw current piece
        if not self.game_over:
            shape, rot, px, py = self.current
            for cx, cy in self.shapes[shape][rot]:
                if py + cy >= 0:
                    self.draw_block(px + cx, py + cy, self.colors[shape])

        if self.game_over:
            self.canvas.create_text(
                self.width // 2,
                self.height // 2,
                text="GAME OVER",
                fill="#f87171",
                font=("Consolas", 24, "bold"),
            )

        info_lines = [
            f"Score : {self.score}",
            f"Lines : {self.lines_cleared}",
            f"Level : {self.level}",
            "",
            f"Next  : {self.next_piece[0]}",
            "",
            "Controls:",
            "←/→ move",
            "↑ rotate",
            "↓ soft drop",
            "Space hard drop",
            "P pause",
            "N new game",
        ]
        if self.paused:
            info_lines.insert(0, "Paused")
        self.info.config(text="\n".join(info_lines))


def main() -> None:
    root = tk.Tk()
    game = TetrisGame(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # Allow clean exit in terminals.
        sys.exit(0)


if __name__ == "__main__":
    main()

