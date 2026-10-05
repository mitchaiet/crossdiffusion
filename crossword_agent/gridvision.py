"""Detect a crossword grid's layout (black squares) from a screenshot with plain image processing.

Vision-language models transcribe clues well but spend minutes of reasoning on grid geometry
and still miscount squares. The geometry is easy for pixels: the grid is the largest dark
connected component (border + cell lines + black squares); its bounding box divided by the
grid size gives the cells; a cell is black if its interior is dark.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass
class GridDetection:
    rows: list[str]
    bbox: tuple[int, int, int, int]  # x0, y0, x1, y1 in the analysed image
    width: int
    height: int
    cell: float
    confidence: float  # fraction of cells whose darkness is decisive (far from the threshold)


def _largest_dark_component(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Bounding box of the connected dark component with the largest bounding-box area."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    best, best_area = None, 0
    label = 0
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs, strict=True):
        if labels[sy, sx]:
            continue
        label += 1
        stack = [(sy, sx)]
        labels[sy, sx] = label
        x0 = x1 = sx
        y0 = y1 = sy
        n = 0
        while stack:
            y, x = stack.pop()
            n += 1
            x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    stack.append((ny, nx))
        area = (x1 - x0 + 1) * (y1 - y0 + 1)
        if area > best_area and n > 50:
            best, best_area = (x0, y0, x1, y1), area
    return best


def _infer_size(dark: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    """Grid lines show up as peaks in the dark-pixel projection; the median spacing between
    peaks is the cell size (robust to lines hidden by black squares), and size = span / cell."""
    x0, y0, x1, y1 = bbox
    sub = dark[y0 : y1 + 1, x0 : x1 + 1]

    def cells_along(frac: np.ndarray, span: int) -> int:
        thr = max(0.5, float(np.median(frac)) + 0.25)
        peaks = []
        on = frac > thr
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                peaks.append((i + j) / 2)
                i = j + 1
            else:
                i += 1
        if len(peaks) < 3:
            return 0
        gaps = np.diff(peaks)
        gaps = gaps[gaps > 4]
        if len(gaps) == 0:
            return 0
        cell = float(np.median(gaps))
        return int(round(span / cell))

    w = cells_along(sub.mean(axis=0), sub.shape[1])
    h = cells_along(sub.mean(axis=1), sub.shape[0])
    return w, h


def _dark_mask(a: np.ndarray) -> np.ndarray:
    """Dark pixels relative to the local paper brightness, so yellowed or unevenly lit scans work
    as well as clean screenshots. The background is a heavy blur of the image."""
    from PIL import ImageFilter

    bg = np.asarray(Image.fromarray((a * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(25)), dtype=np.float32) / 255.0
    return (a < bg - 0.18) | (a < 0.35)


def detect_grid(image_bytes: bytes, width: int | None = None, height: int | None = None,
                analysis_px: int = 1400) -> GridDetection | None:
    img = Image.open(io.BytesIO(image_bytes)).convert("L")
    scale = analysis_px / max(img.size)
    if scale < 1.0 or scale > 1.2:  # shrink big screenshots; enlarge small scans so 1px lines survive
        img = img.resize((int(img.width * min(scale, 2.5)), int(img.height * min(scale, 2.5))), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    base_dark = _dark_mask(a)
    result = None
    # first pass on the raw mask; if the line structure is broken (low-res scans), dilate the
    # mask so 1-pixel lines reconnect and try again
    for dilate in (0, 1, 2):
        dark = base_dark
        for _ in range(dilate):
            d = dark.copy()
            d[1:, :] |= dark[:-1, :]
            d[:-1, :] |= dark[1:, :]
            d[:, 1:] |= dark[:, :-1]
            d[:, :-1] |= dark[:, 1:]
            dark = d
        bbox = _largest_dark_component(dark)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        if bw < 60 or bh < 60:
            continue
        w_guess, h_guess = (width, height) if (width and height) else _infer_size(dark, bbox)
        if not w_guess or not h_guess or w_guess > 40 or h_guess > 40 or min(w_guess, h_guess) < 5:
            continue
        if max(bw, bh) < 0.2 * min(a.shape):  # a real grid is a substantial part of the picture
            continue
        cw, ch = bw / w_guess, bh / h_guess
        if abs(cw - ch) / max(cw, ch) > 0.25:
            continue
        result = (bbox, w_guess, h_guess)
        break
    if result is None:
        return None  # could not read the line structure (blurred / skewed photo)
    (x0, y0, x1, y1), width, height = result
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    cw, ch = bw / width, bh / height
    if abs(cw - ch) / max(cw, ch) > 0.25:  # cells should be roughly square
        return None
    rows: list[str] = []
    decisive = 0
    for r in range(height):
        row = ""
        for c in range(width):
            # sample the inner 50% of the cell so lines and numbers do not count
            cx0 = int(x0 + (c + 0.25) * cw)
            cx1 = int(x0 + (c + 0.75) * cw)
            cy0 = int(y0 + (r + 0.25) * ch)
            cy1 = int(y0 + (r + 0.75) * ch)
            patch = a[cy0 : max(cy1, cy0 + 1), cx0 : max(cx1, cx0 + 1)]
            darkness = 1.0 - float(patch.mean())
            row += "#" if darkness > 0.5 else "."
            if abs(darkness - 0.5) > 0.3:
                decisive += 1
        rows.append(row)
    return GridDetection(rows, bbox, width, height, (cw + ch) / 2, decisive / (width * height))
