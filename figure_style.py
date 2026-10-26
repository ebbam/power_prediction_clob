"""Shared publication style: every figure is drawn at its printed width with one type scale,
so text renders at identical point sizes across figures when they are placed at full width.

Import and call apply() before drawing, build figures PAGE_WIDTH (full width) or COLUMN_WIDTH
(half-page) inches wide, and save them with save(), which checks fonts and page bounds.
"""

import warnings

import matplotlib as mpl
from matplotlib.text import Text
from matplotlib.transforms import Bbox


PAGE_WIDTH = 7.0          # inches; full text width (~178 mm), for Figure 1
COLUMN_WIDTH = 3.4        # inches; one column (~86 mm), for the half-page figures
WIDTHS = {PAGE_WIDTH, COLUMN_WIDTH}

PANEL = 10                # panel letters (bold)
TITLE = 9                 # panel titles
LABEL = 8                 # axis labels
SMALL = 7                 # tick labels, legends, in-plot notes
SIZES = {PANEL, TITLE, LABEL, SMALL}


def apply():
    mpl.rcParams.update({
        "font.size": SMALL,
        "axes.titlesize": TITLE,
        "axes.labelsize": LABEL,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "legend.title_fontsize": SMALL,
        "figure.titlesize": TITLE,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.0,
        "patch.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "boxplot.boxprops.linewidth": 0.6,
        "boxplot.whiskerprops.linewidth": 0.6,
        "boxplot.capprops.linewidth": 0.6,
        "boxplot.medianprops.linewidth": 0.8,
        "savefig.dpi": 300,
        "figure.constrained_layout.w_pad": 0.06,   # inches of margin round the page
        "figure.constrained_layout.h_pad": 0.06,
    })


def check_fonts(figure, name):
    """Raise if any visible text is not one of the standard sizes, falls outside the page,
    or the figure is not one of the standard widths."""
    width = figure.get_size_inches()[0]
    if not any(abs(width - standard) < 1e-6 for standard in WIDTHS):
        raise ValueError(f"{name}: figure is {width:g} in wide, not one of {sorted(WIDTHS)}")
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # a collapsed constrained layout must fail loudly
        figure.canvas.draw()
    hidden = set()      # tick labels outside an axis's view are kept but never drawn
    for ax in figure.get_axes():
        for axis in (ax.xaxis, ax.yaxis):
            low, high = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not low - 1e-9 * max(1, abs(high)) <= tick.get_loc() <= high * (1 + 1e-9) + 1e-12:
                    hidden.update((id(tick.label1), id(tick.label2)))
    texts = [text for text in figure.findobj(Text)
             if text.get_visible() and text.get_text().strip() and id(text) not in hidden]
    stray = sorted({round(text.get_fontsize(), 2) for text in texts} - SIZES)
    if stray:
        raise ValueError(f"{name}: text at non-standard sizes {stray} pt")
    # Saved at exactly the page size, so nothing may fall outside the canvas.
    renderer = figure.canvas.get_renderer()
    figure.canvas.draw()
    page = figure.bbox.padded(-3)       # a small margin: text may not touch the edge
    outside = [text.get_text()[:30] for text in texts
               if not page.contains(*text.get_window_extent(renderer).p0)
               or not page.contains(*text.get_window_extent(renderer).p1)]
    if outside:
        raise ValueError(f"{name}: text outside the page: {outside[:5]}")
    boxes = [(text.get_text()[:25], text.get_window_extent(renderer).padded(-1))
             for text in texts]
    clashes = [(a, b) for i, (a, box_a) in enumerate(boxes) for b, box_b in boxes[i + 1:]
               if box_a.overlaps(box_b)]
    if clashes:
        raise ValueError(f"{name}: overlapping text: {clashes[:4]}")


def save(figure, path_without_suffix, name):
    """Check, then save PDF and PNG at exactly the figure's size (no tight cropping, which
    would change the printed width and so the font sizes)."""
    check_fonts(figure, name)
    figure.savefig(f"{path_without_suffix}.pdf")
    figure.savefig(f"{path_without_suffix}.png", dpi=300)


def label_panels(axes_and_letters):
    """Panel letters at the left edge of each panel's labels, level with its title.

    Placed after the layout is fixed, from each axis's drawn extent, so they never collide
    with tick or axis labels at any figure width.
    """
    pairs = list(axes_and_letters)
    if not pairs:
        return
    figure = pairs[0][0].get_figure(root=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # a collapsed layout must not pass silently
        figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    for ax, letter in pairs:
        # Left edge of the axis and its y-axis labels (not legends, which can be wider).
        yaxis = ax.yaxis.get_tightbbox(renderer)       # None when the y-axis has no labels
        left = min(ax.bbox.x0, yaxis.x0) if yaxis is not None else ax.bbox.x0
        box = Bbox.from_extents(left, 0, left + 1, 1).transformed(
            figure.transFigure.inverted())
        top = ax.get_position().y1
        holder = ax.get_figure()
        if holder is not figure and getattr(holder, "_suptitle", None) is not None:
            top = holder._suptitle.get_window_extent(renderer).transformed(
                figure.transFigure.inverted()).y1
        elif ax.get_title():
            top = ax.title.get_window_extent(renderer).transformed(
                figure.transFigure.inverted()).y1
        figure.text(box.x0, top, letter, fontsize=PANEL, fontweight="bold",
                    ha="left", va="top")
    figure.set_layout_engine("none")      # keep the measured positions when saving
