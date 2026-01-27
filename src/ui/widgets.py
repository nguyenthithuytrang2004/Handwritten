import tkinter as tk


def _rounded_rect_coords(w, h, r):
    # returns coords for rectangle and arcs (not used directly)
    return


class RoundedFrame(tk.Frame):
    """
    A Frame with a rounded-rectangle background drawn on a Canvas.
    Child widgets should be placed inside the `inner` Frame attribute.
    """
    def __init__(self, parent, bg="#fff9e6", border="#e0e0e0", radius=12, padding=8, **kwargs):
        # Determine a safe background color to give the outer Frame.
        if "bg" in kwargs:
            parent_bg = kwargs.get("bg")
        else:
            parent_bg = None
            try:
                parent_bg = parent.cget("background")
            except Exception:
                try:
                    parent_bg = parent.cget("bg")
                except Exception:
                    parent_bg = None

        super().__init__(parent, bg=parent_bg if parent_bg is not None else bg)
        self._bg = bg
        self._border = border
        self._radius = radius
        self._padding = padding

        self._canvas = tk.Canvas(self, highlightthickness=0, bg=self["bg"])
        self._canvas.pack(fill=tk.BOTH, expand=True)

        # inner frame where user places widgets
        self.inner = tk.Frame(self._canvas, bg=self._bg)
        self._win = self._canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self._canvas.bind("<Configure>", self._on_configure)

    def _draw_rounded_rect(self, w, h):
        self._canvas.delete("bg_round")
        r = max(0, min(self._radius, int(min(w, h) / 2)))
        # corners as arcs
        x0, y0, x1, y1 = 2, 2, w - 2, h - 2
        # create rounded rectangle via polygons and arcs
        try:
            # background
            self._canvas.create_rectangle(x0 + r, y0, x1 - r, y1, fill=self._bg, outline="", tags="bg_round")
            self._canvas.create_rectangle(x0, y0 + r, x1, y1 - r, fill=self._bg, outline="", tags="bg_round")
            # four corner arcs
            self._canvas.create_oval(x0, y0, x0 + 2 * r, y0 + 2 * r, fill=self._bg, outline="", tags="bg_round")
            self._canvas.create_oval(x1 - 2 * r, y0, x1, y0 + 2 * r, fill=self._bg, outline="", tags="bg_round")
            self._canvas.create_oval(x0, y1 - 2 * r, x0 + 2 * r, y1, fill=self._bg, outline="", tags="bg_round")
            self._canvas.create_oval(x1 - 2 * r, y1 - 2 * r, x1, y1, fill=self._bg, outline="", tags="bg_round")
            # border (thin)
            self._canvas.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, style="arc", outline=self._border, tags="bg_round")
            self._canvas.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, style="arc", outline=self._border, tags="bg_round")
            self._canvas.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, style="arc", outline=self._border, tags="bg_round")
            self._canvas.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, style="arc", outline=self._border, tags="bg_round")
        except Exception:
            # fallback: simple rectangle
            self._canvas.create_rectangle(x0, y0, x1, y1, fill=self._bg, outline=self._border, tags="bg_round")

    def _on_configure(self, event):
        try:
            w = event.width
            h = event.height
            # Resize inner window to leave padding
            pad = int(self._padding)
            self._canvas.coords(self._win, pad, pad)
            self._canvas.itemconfig(self._win, width=max(1, w - 2 * pad), height=max(1, h - 2 * pad))
            self._draw_rounded_rect(w, h)
        except Exception:
            pass


class RoundedLabelFrame(RoundedFrame):
    """
    Rounded frame with a title label at the top-left to mimic LabelFrame.
    Use `.inner` to place child widgets as with regular Frame.
    """
    def __init__(self, parent, text="", font=None, **kwargs):
        super().__init__(parent, **kwargs)
        # create title label inside canvas above the inner frame
        self._title_var = tk.StringVar(value=text)
        self._title = tk.Label(self._canvas, text=text, bg=self._bg, fg=kwargs.get("fg", "#1b5e20"), font=font)
        # place title; will be repositioned on configure
        self._title_win = self._canvas.create_window((12, -8), window=self._title, anchor="nw", tags="bg_round")
        # ensure title sits above background
        self._canvas.tag_raise(self._title_win)

    def _on_configure(self, event):
        super()._on_configure(event)
        try:
            # keep title near top-left with small offset
            self._canvas.coords(self._title_win, int(max(12, self._padding)), -int(self._padding / 2))
        except Exception:
            pass


