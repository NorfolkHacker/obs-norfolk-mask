#!/usr/bin/env python3
"""Generate the Norfolk Hacker Mask artwork (original art) as transparent PNGs.

Usage: python3 art/make_art.py data/

Canvas sizes and anchors MUST match src/layout.hpp (namespace nhm::art).
"""
import math
import os
import sys

import cairo

PURPLE = (0.69, 0.15, 1.0)
PLATE_W, PLATE_H = 512, 600
ANCHOR = (256, 240)  # cyclops socket centre; 180 px = one inter-eye distance
HEAD_W, HEAD_H = 160, 260
HEAD_ANCHOR = (80, 135)
TAU = 2 * math.pi


def rounded_rect(c, x, y, w, h, r):
    c.new_sub_path()
    c.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    c.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    c.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    c.arc(x + r, y + r, r, math.pi, 1.5 * math.pi)
    c.close_path()


def neon_stroke(c, path_fn, color, width):
    """Stroke a path several times, wide+faint to thin+bright, then a hot core."""
    for w, a in ((width * 5, 0.07), (width * 3, 0.14), (width * 1.8, 0.35), (width, 1.0)):
        c.new_path()
        path_fn(c)
        c.set_source_rgba(*color, a)
        c.set_line_width(w)
        c.stroke()
    c.new_path()
    path_fn(c)
    c.set_source_rgba(1.0, 0.85, 1.0, 0.8)
    c.set_line_width(width * 0.35)
    c.stroke()


def plate_outline(c):
    cx = 256
    c.move_to(cx, 18)
    c.curve_to(380, 18, 470, 80, 478, 190)    # forehead -> image-right temple
    c.curve_to(486, 280, 470, 360, 420, 440)  # cheek
    c.curve_to(380, 510, 330, 575, cx, 585)   # jaw -> chin
    c.curve_to(182, 575, 132, 510, 92, 440)   # chin -> other jaw
    c.curve_to(42, 360, 26, 280, 34, 190)
    c.curve_to(42, 80, 132, 18, cx, 18)
    c.close_path()


def mirror_x(side):
    return lambda x: 256 + side * (x - 256)


def make_plate(path):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, PLATE_W, PLATE_H)
    c = cairo.Context(s)

    # Brushed-metal body
    g = cairo.LinearGradient(0, 18, 0, 585)
    g.add_color_stop_rgb(0.0, 0.86, 0.87, 0.92)
    g.add_color_stop_rgb(0.45, 0.62, 0.64, 0.72)
    g.add_color_stop_rgb(1.0, 0.30, 0.31, 0.38)
    plate_outline(c)
    c.set_source(g)
    c.fill_preserve()
    # Edge darkening for roundness, tinted purple
    r = cairo.RadialGradient(256, 260, 120, 256, 260, 300)
    r.add_color_stop_rgba(0, 0, 0, 0, 0)
    r.add_color_stop_rgba(1, 0.08, 0.02, 0.15, 0.65)
    c.set_source(r)
    c.fill()

    c.save()
    plate_outline(c)
    c.clip()
    # Panel seams
    c.set_source_rgba(0.12, 0.10, 0.16, 0.9)
    c.set_line_width(4)
    c.move_to(40, 170)
    c.curve_to(150, 140, 362, 140, 472, 170)  # brow seam
    c.stroke()
    for side in (1, -1):
        X = mirror_x(side)
        c.move_to(X(330), 300)
        c.line_to(X(360), 420)
        c.line_to(X(330), 560)
        c.stroke()
        c.move_to(X(478), 300)
        c.line_to(X(372), 330)
        c.stroke()
    # Nose ridge highlight
    c.set_source_rgba(1, 1, 1, 0.35)
    c.set_line_width(3)
    c.move_to(256, 322)
    c.line_to(256, 410)
    c.stroke()
    # Mouth grille
    c.set_source_rgb(0.10, 0.08, 0.13)
    rounded_rect(c, 176, 420, 160, 110, 26)
    c.fill()
    for i in range(5):
        c.set_source_rgba(0.75, 0.75, 0.82, 1)
        rounded_rect(c, 190, 434 + i * 19, 132, 8, 4)
        c.fill()
    c.restore()

    # Neon outline and circuit traces
    neon_stroke(c, plate_outline, PURPLE, 4)
    for side in (1, -1):
        X = mirror_x(side)

        def trace_a(ctx, X=X):
            ctx.move_to(X(345), 250)
            ctx.line_to(X(420), 250)
            ctx.line_to(X(445), 290)
            ctx.line_to(X(445), 360)

        def trace_b(ctx, X=X):
            ctx.move_to(X(318), 300)
            ctx.line_to(X(395), 380)
            ctx.line_to(X(395), 470)

        def trace_brow(ctx, X=X):
            ctx.move_to(X(300), 120)
            ctx.line_to(X(380), 120)
            ctx.line_to(X(410), 95)

        for fn in (trace_a, trace_b, trace_brow):
            neon_stroke(c, fn, PURPLE, 2.5)
        for nx, ny in ((445, 360), (395, 470), (410, 95)):
            neon_stroke(c, lambda ctx, nx=nx, ny=ny, X=X: ctx.arc(X(nx), ny, 5, 0, TAU), PURPLE, 2)

    # Cyclops socket: metal rim, dark hole, neon ring, dim lens
    ax, ay = ANCHOR
    rim = cairo.RadialGradient(ax, ay - 20, 10, ax, ay, 80)
    rim.add_color_stop_rgb(0, 0.95, 0.95, 1.0)
    rim.add_color_stop_rgb(1, 0.35, 0.36, 0.44)
    c.new_path()
    c.arc(ax, ay, 78, 0, TAU)
    c.set_source(rim)
    c.fill()
    c.arc(ax, ay, 62, 0, TAU)
    c.set_source_rgb(0.05, 0.02, 0.08)
    c.fill()
    neon_stroke(c, lambda ctx: ctx.arc(ax, ay, 68, 0, TAU), PURPLE, 4)
    lens = cairo.RadialGradient(ax, ay, 0, ax, ay, 55)
    lens.add_color_stop_rgb(0.0, 0.90, 0.60, 1.0)
    lens.add_color_stop_rgb(0.35, 0.55, 0.10, 0.90)
    lens.add_color_stop_rgb(1.0, 0.12, 0.02, 0.20)
    c.new_path()
    c.arc(ax, ay, 55, 0, TAU)
    c.set_source(lens)
    c.fill()
    # Rim bolts
    for deg in (45, 135, 225, 315):
        a = math.radians(deg)
        c.new_path()
        c.arc(ax + 74 * math.cos(a), ay + 74 * math.sin(a), 4, 0, TAU)
        c.set_source_rgb(0.25, 0.25, 0.3)
        c.fill()

    s.write_to_png(path)


def make_headphone(path, mirror):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, HEAD_W, HEAD_H)
    c = cairo.Context(s)
    if mirror:
        c.translate(HEAD_W, 0)
        c.scale(-1, 1)
    hx, hy = HEAD_ANCHOR
    # Band stub
    c.set_source_rgb(0.12, 0.10, 0.16)
    rounded_rect(c, 62, 0, 36, 60, 10)
    c.fill()
    # Cup body
    g = cairo.LinearGradient(22, 0, 138, 0)
    g.add_color_stop_rgb(0, 0.24, 0.22, 0.30)
    g.add_color_stop_rgb(1, 0.07, 0.05, 0.10)
    rounded_rect(c, 22, 40, 116, 190, 50)
    c.set_source(g)
    c.fill()

    def ring(ctx):
        ctx.save()
        ctx.translate(hx, hy)
        ctx.scale(1, 1.45)
        ctx.arc(0, 0, 38, 0, TAU)
        ctx.restore()

    neon_stroke(c, ring, PURPLE, 4)
    core = cairo.RadialGradient(hx, hy, 0, hx, hy, 24)
    core.add_color_stop_rgba(0, 1.0, 0.85, 1.0, 1)
    core.add_color_stop_rgba(0.4, *PURPLE, 0.9)
    core.add_color_stop_rgba(1, *PURPLE, 0)
    c.new_path()
    c.arc(hx, hy, 24, 0, TAU)
    c.set_source(core)
    c.fill()
    s.write_to_png(path)


MILL_W, MILL_H = 256, 512
MILL_HUB = (128, 112)
SAILS_SIZE = 512
SILHOUETTE = (0.06, 0.02, 0.11)


def tower_outline(c):
    c.move_to(80, 150)    # top-left of tower under the cap
    c.line_to(176, 150)
    c.line_to(212, 500)
    c.line_to(44, 500)
    c.close_path()


def cap_outline(c):
    c.move_to(66, 156)
    c.curve_to(66, 92, 190, 92, 190, 156)
    c.close_path()


def make_windmill_body(path):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, MILL_W, MILL_H)
    c = cairo.Context(s)
    # Tower with a faint purple sheen toward the sky side
    g = cairo.LinearGradient(44, 0, 212, 0)
    g.add_color_stop_rgb(0, 0.10, 0.04, 0.18)
    g.add_color_stop_rgb(1, *SILHOUETTE)
    tower_outline(c)
    c.set_source(g)
    c.fill()
    # Cap and fantail (fan on a frame at the back of the cap)
    cap_outline(c)
    c.set_source_rgb(*SILHOUETTE)
    c.fill()
    c.set_source_rgb(*SILHOUETTE)
    c.set_line_width(4)
    c.move_to(176, 140)
    c.line_to(226, 168)
    c.stroke()
    for k in range(6):
        a = k * TAU / 6
        c.move_to(226, 168)
        c.line_to(226 + 18 * math.cos(a), 168 + 18 * math.sin(a))
        c.stroke()
    # Gallery rail, door and lit windows
    c.set_line_width(5)
    c.move_to(58, 300)
    c.line_to(198, 300)
    c.stroke()
    c.set_source_rgb(0.02, 0.0, 0.04)
    rounded_rect(c, 112, 440, 32, 60, 14)
    c.fill()
    for wx, wy in ((128, 220), (112, 360), (146, 360)):
        c.set_source_rgba(*PURPLE, 0.9)
        rounded_rect(c, wx - 7, wy - 10, 14, 20, 5)
        c.fill()
    # Neon rim
    neon_stroke(c, tower_outline, PURPLE, 2.5)
    neon_stroke(c, cap_outline, PURPLE, 2.5)
    # Hub boss (sails are drawn separately, rotating around MILL_HUB)
    hx, hy = MILL_HUB
    c.new_path()
    c.arc(hx, hy, 10, 0, TAU)
    c.set_source_rgb(*SILHOUETTE)
    c.fill()
    s.write_to_png(path)


def make_windmill_sails(path):
    s = cairo.ImageSurface(cairo.FORMAT_ARGB32, SAILS_SIZE, SAILS_SIZE)
    c = cairo.Context(s)
    cx = cy = SAILS_SIZE / 2
    c.translate(cx, cy)
    for k in range(4):
        c.save()
        c.rotate(k * TAU / 4)
        # Stock (arm) from the hub outwards along -y
        c.set_source_rgb(*SILHOUETTE)
        rounded_rect(c, -5, -240, 10, 240, 4)
        c.fill()

        def frame(ctx):
            ctx.rectangle(5, -236, 44, 180)

        # Lattice: sail frame on one side of the stock with bars across
        c.set_source_rgba(*SILHOUETTE, 0.85)
        frame(c)
        c.fill()
        c.set_source_rgba(0.35, 0.10, 0.55, 0.9)
        c.set_line_width(2)
        for i in range(1, 9):
            y = -236 + i * 20
            c.move_to(5, y)
            c.line_to(49, y)
            c.stroke()
        c.move_to(27, -236)
        c.line_to(27, -56)
        c.stroke()
        neon_stroke(c, frame, PURPLE, 2)
        c.restore()
    c.new_path()
    c.arc(0, 0, 12, 0, TAU)
    c.set_source_rgb(*SILHOUETTE)
    c.fill()
    neon_stroke(c, lambda ctx: ctx.arc(0, 0, 12, 0, TAU), PURPLE, 2)
    s.write_to_png(path)


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(out, exist_ok=True)
    make_plate(os.path.join(out, "faceplate.png"))
    make_headphone(os.path.join(out, "headphone_l.png"), mirror=False)
    make_headphone(os.path.join(out, "headphone_r.png"), mirror=True)
    make_windmill_body(os.path.join(out, "windmill_body.png"))
    make_windmill_sails(os.path.join(out, "windmill_sails.png"))


if __name__ == "__main__":
    main()
