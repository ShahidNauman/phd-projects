"""Tkinter-based Digit Collector Application.

This module provides a standalone graphical user interface (GUI) application
designed for interactive drawing and collection of 16x16 grayscale handwritten
digit samples.

Architecture Overview:
    - Main Application: The `DigitCollectorApp` class initializes a Tkinter window,
      instantiates the layout, and manages state.
    - View Layer: A zoomed `tk.Canvas` displays the drawing surface. Each cell
      representing a pixel is rendered using a persistent canvas rectangle to allow
      efficient state-based color changes.
    - Controller/Events Layer: Binds click `<Button-1>` and drag `<B1-Motion>`
      handlers to record draw states.
    - Data Layer: Maintains a 2D Python nested list (`List[List[int]]`) matching the
      active drawing.

Assumptions:
    - Grayscale ranges are binary for this collection pad: 0 = Black, 255 = White.
    - A 10x scale zoom factor maps 16x16 logical matrices to a 160x160 user drawing space.
    - Standard system mouse events capture coordinates relative to the canvas origin.

Validation Logic:
    - Out-of-bounds guards prevent mouse coordinates outside the 160x160 canvas viewport
      from triggering index updates on the 16x16 array.
"""

import tkinter as tk
from typing import List

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------
GRID_SIZE: int = 16  # Number of pixels along each axis of the logical image
ZOOM: int = 10  # Display scale factor (logical pixel -> canvas pixels)

# Derived canvas dimensions
CANVAS_SIZE: int = GRID_SIZE * ZOOM  # 160 px

# Color values
BLACK: int = 0  # Background pixel value
WHITE: int = 255  # Drawn pixel value

# Hex colors used by the Tk Canvas
HEX_BLACK: str = "#000000"
HEX_WHITE: str = "#ffffff"
HEX_GRID: str = "#333333"  # Subtle dark-grey grid lines


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class DigitCollectorApp:
    """Tkinter application for drawing 16x16 handwritten digit samples.

    Attributes:
        root: The top-level Tkinter root window.
        canvas: The enlarged drawing surface.
        pixel_data: 16x16 list of grayscale values (0 or 255).
        rects: Canvas item IDs of every grid cell rectangle, used for efficient
            color updates without clearing the canvas.
    """

    def __init__(self, root: tk.Tk) -> None:
        """Initializes the application window, canvas, data structures, and bindings.

        Args:
            root: The Tk root window passed in from the entry point.
        """
        self.root: tk.Tk = root
        self.root.title("Digit Collector - 16x16 Drawing Pad")
        self.root.resizable(False, False)

        # Internal image representation: 16x16 grid, starts completely black (0)
        self.pixel_data: List[List[int]] = self._make_blank_grid()

        # Canvas item ID lookup table: rects[row][col] -> Tk item ID (integer)
        # Populated inside _build_canvas()
        self.rects: List[List[int]] = []

        # Build UI layout elements
        self._build_canvas()
        self._build_controls()
        self._bind_mouse()

    # -----------------------------------------------------------------------
    # UI construction helpers
    # -----------------------------------------------------------------------

    def _build_canvas(self) -> None:
        """Creates the Tk Canvas and draws the initial black grid with grid lines."""
        # Add a thin outer frame for aesthetic margins and padding
        frame = tk.Frame(self.root, bg="#1a1a1a", padx=2, pady=2)
        frame.pack(side=tk.TOP, padx=10, pady=(10, 4))

        self.canvas = tk.Canvas(
            frame,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg=HEX_BLACK,
            cursor="crosshair",
            highlightthickness=0,  # Suppresses default Tk focus ring highlight
        )
        self.canvas.pack()

        # Draw individual rectangles to act as interactive cells
        for row in range(GRID_SIZE):
            row_ids: List[int] = []
            for col in range(GRID_SIZE):
                # Calculate bounding box coordinates based on zoom factor
                x0 = col * ZOOM
                y0 = row * ZOOM
                # Subtract 1 pixel to reserve space for the grid lines
                x1 = x0 + ZOOM - 1
                y1 = y0 + ZOOM - 1

                # Instantiate filled rectangle object
                item_id = self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=HEX_BLACK,
                    outline="",  # Individual cells have no borders; drawn globally instead
                )
                row_ids.append(item_id)
            self.rects.append(row_ids)

        # Draw grid lines on top of rectangles for structural guidance
        # Vertical grid separators
        for col in range(1, GRID_SIZE):
            x = col * ZOOM
            self.canvas.create_line(x, 0, x, CANVAS_SIZE, fill=HEX_GRID)

        # Horizontal grid separators
        for row in range(1, GRID_SIZE):
            y = row * ZOOM
            self.canvas.create_line(0, y, CANVAS_SIZE, y, fill=HEX_GRID)

    def _build_controls(self) -> None:
        """Creates the control panel with Clear, Export Array, and Exit buttons."""
        bar = tk.Frame(self.root, bg="#1a1a1a")
        bar.pack(side=tk.TOP, fill=tk.X, padx=10, pady=(0, 10))

        # Standard button styling properties
        btn_style = {
            "font": ("Consolas", 10, "bold"),
            "width": 14,
            "cursor": "hand2",
            "pady": 6,
        }

        # Clear button setup
        tk.Button(
            bar,
            text="Clear",
            bg="#2d2d2d",
            fg="#ffffff",
            activebackground="#444444",
            activeforeground="#ffffff",
            command=self._clear_canvas,
            relief=tk.FLAT,
            **btn_style,
        ).pack(side=tk.LEFT, padx=(0, 6))

        # Export Array button setup
        tk.Button(
            bar,
            text="Export Array",
            bg="#1a5276",
            fg="#ffffff",
            activebackground="#2e86c1",
            activeforeground="#ffffff",
            command=self._export_array,
            relief=tk.FLAT,
            **btn_style,
        ).pack(side=tk.LEFT, padx=(0, 6))

        # Exit button setup
        tk.Button(
            bar,
            text="Exit",
            bg="#641e16",
            fg="#ffffff",
            activebackground="#922b21",
            activeforeground="#ffffff",
            command=self.root.destroy,
            relief=tk.FLAT,
            **btn_style,
        ).pack(side=tk.LEFT)

    def _bind_mouse(self) -> None:
        """Binds mouse click and drag gestures to the interactive drawing handler."""
        # Binding both actions prevents draw gaps during fast cursor movements
        self.canvas.bind("<Button-1>", self._on_mouse_paint)
        self.canvas.bind("<B1-Motion>", self._on_mouse_paint)

    # -----------------------------------------------------------------------
    # Event handlers
    # -----------------------------------------------------------------------

    def _on_mouse_paint(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        """Paints the grid cell currently targeted by the mouse white.

        Args:
            event: The Tkinter event object containing the cursor's coordinate context.
        """
        # Map raw screen coordinates (0-159) to cell matrix index positions (0-15)
        col = event.x // ZOOM
        row = event.y // ZOOM

        # Validation Guard: Prevent IndexErrors if cursor drifts outside canvas boundaries
        if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
            return

        # Redraw only when state changes to avoid redundant Tk rendering calls
        if self.pixel_data[row][col] != WHITE:
            self.pixel_data[row][col] = WHITE
            self.canvas.itemconfig(self.rects[row][col], fill=HEX_WHITE)

    # -----------------------------------------------------------------------
    # Button callbacks
    # -----------------------------------------------------------------------

    def _clear_canvas(self) -> None:
        """Resets all logical matrix values and canvas colors to black (0)."""
        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                if self.pixel_data[row][col] != BLACK:
                    self.pixel_data[row][col] = BLACK
                    self.canvas.itemconfig(self.rects[row][col], fill=HEX_BLACK)

        print("\n[Canvas cleared]")

    def _export_array(self) -> None:
        """Outputs the drawn 16x16 array matrix and pixel counts to the console."""
        black_count = 0
        white_count = 0

        print(f"\n{'=' * 52}")
        print(f"Image Shape: {GRID_SIZE} x {GRID_SIZE}")
        print()

        for row in self.pixel_data:
            # Construct a human-readable, spacing-aligned row string
            formatted = ", ".join(f"{v:>3}" for v in row)
            print(f"[{formatted}]")

            # Accumulate totals
            for v in row:
                if v == BLACK:
                    black_count += 1
                else:
                    white_count += 1

        print(f"\nBlack Pixels : {black_count}")
        print(f"White Pixels : {white_count}")
        print(f"{'=' * 52}\n")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _make_blank_grid() -> List[List[int]]:
        """Constructs a raw 16x16 nested list filled with zeros.

        Returns:
            A nested 16x16 integer list.
        """
        return [[BLACK] * GRID_SIZE for _ in range(GRID_SIZE)]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Launches the application loop."""
    root = tk.Tk()
    root.configure(bg="#1a1a1a")

    app = DigitCollectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
