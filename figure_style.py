"""Shared publication style: every figure is drawn at its printed width with one type scale,
so text renders at identical point sizes across figures when they are placed at full width.

Import and call apply() before drawing, build figures PAGE_WIDTH (full width) or COLUMN_WIDTH
(half-page) inches wide, and save them with save(), which checks fonts and page bounds.

Every figure takes its colours from PALETTE by name, through color(), at the moment it is
drawn, so changing a colour here (or in the notebook) changes it in every figure.
"""

import warnings

import matplotlib as mpl
from matplotlib.text import Text
from matplotlib.transforms import Bbox


PAGE_WIDTH = 7.0          # inches; full text width (~178 mm)
COLUMN_WIDTH = 3.4        # inches; one column (~86 mm), for the half-page figures
WIDTHS = {PAGE_WIDTH, COLUMN_WIDTH}

PANEL = 10                # panel letters (bold)
TITLE = 9                 # panel titles
LABEL = 8                 # axis labels
SMALL = 7                 # tick labels, legends, in-plot notes
SIZES = {PANEL, TITLE, LABEL, SMALL}

PALETTE = {
    # The power_prediction palette (cell 3 of the original notebooks).
    "lavender": "#CC99FF", "navy": "#000080", "green": "#339966", "orange": "#F4A261",
    "ruby": "#B22222", "violet": "#6A4C93", "indigo": "#1D3557", "teal": "#2A9D8F",
    "steel": "#457B9D", "amber": "#FFB347", "crimson": "#E63946",
    # Further colours used by single figures.
    "lilac": "#B788E5", "sky": "#1E90FF", "tangerine": "#FF8C00", "royal": "#4169E1",
    "aqua": "#20B2AA", "slate": "#6A5ACD", "plum": "#8E3B8E", "gold": "#FFA500",
    "deep_plum": "#5B1F7A", "burnt": "#C45A00",
}


def color(names):
    """The PALETTE colour for a name, or a list of colours for a sequence of names."""
    return PALETTE[names] if isinstance(names, str) else [PALETTE[name] for name in names]


def apply():
    mpl.rcParams.update({
        "font.size": SMALL,
        "axes.titlesize": TITLE,
        "axes.labelsize": LABEL,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "legend.title_fontsize": SMALL,
        # One legend style everywhere: a thin grey square frame on white.
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 1.0,
        "legend.edgecolor": "#BBBBBB",
        "legend.handlelength": 1.4,
        "legend.columnspacing": 1.0,
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


def label_panels(axes_and_letters, gap=4, align_tolerance=0.1, lift=0):
    """Bold panel letters level with each panel's title (or its subfigure's title when it has
    none), at the left edge of its labels, raised by `lift` points.

    Letters whose positions lie within align_tolerance of each other (a share of the figure's
    width or height, chained, so a whole column counts together) are aligned: one x for
    letters in the same column, one height for letters in the same row. Each letter also
    keeps `gap` points clear of its panel's title. Placed after the layout is fixed, from each
    axis's drawn extent, so they never collide with tick or axis labels at any figure width.
    """
    pairs = list(axes_and_letters)
    if not pairs:
        return
    figure = pairs[0][0].get_figure(root=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")       # a collapsed layout must not pass silently
        figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    to_figure = figure.transFigure.inverted()
    gap = gap / 72 / figure.get_size_inches()[0]       # points -> share of figure width
    margin = 0.01

    letters, lefts, tops, ceilings = [], [], [], []
    for ax, letter in pairs:
        text = figure.text(0, 0, letter, fontsize=PANEL, fontweight="bold", ha="left", va="top")
        width = text.get_window_extent(renderer).transformed(to_figure).width
        # Left edge of the axis and its y-axis labels (not legends, which can be wider).
        yaxis = ax.yaxis.get_tightbbox(renderer)       # None when the y-axis has no labels
        left = Bbox.from_extents(min(ax.bbox.x0, yaxis.x0) if yaxis else ax.bbox.x0, 0,
                                 ax.bbox.x0 + 1, 1).transformed(to_figure).x0
        top = ax.get_position().y1
        holder = ax.get_figure()
        suptitle = getattr(holder, "_suptitle", None) if holder is not figure else None
        # A letter level with its own title must stay below its subfigure's title.
        ceilings.append(suptitle.get_window_extent(renderer).transformed(to_figure).y0
                        - gap * figure.get_size_inches()[0] / figure.get_size_inches()[1]
                        if suptitle is not None and ax.get_title() else 1.0)
        if ax.get_title():
            title = ax.title.get_window_extent(renderer).transformed(to_figure)
            top = title.y1
            left = min(left, title.x0 - gap - width)    # keep clear of the title
        elif suptitle is not None:
            top = suptitle.get_window_extent(renderer).transformed(to_figure).y1
        letters.append(text)
        lefts.append(max(left, margin))
        tops.append(top + lift / 72 / figure.get_size_inches()[1])

    def aligned(values, pick):
        """Each value replaced by pick() of its group: values chained within the tolerance."""
        order = sorted(range(len(values)), key=values.__getitem__)
        groups, result = [[order[0]]], list(values)
        for previous, index in zip(order, order[1:]):
            if values[index] - values[previous] < align_tolerance:
                groups[-1].append(index)
            else:
                groups.append([index])
        for group in groups:
            for index in group:
                result[index] = pick(values[member] for member in group)
        return result

    for text, x, y, ceiling in zip(letters, aligned(lefts, min), aligned(tops, max), ceilings):
        text.set_position((x, min(y, ceiling)))
    figure.set_layout_engine("none")      # keep the measured positions when saving


