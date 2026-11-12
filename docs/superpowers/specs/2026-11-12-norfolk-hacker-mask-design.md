# Norfolk Hacker Mask — OBS face-tracking filter

Date: 2026-11-12
Status: design approved in chat, awaiting spec review

## Goal

A native OBS Studio plugin that adds a video filter, **"Norfolk Hacker Mask"**,
to a webcam source. It tracks the person's face and draws an original
purple-neon cyborg mask over it, styled after the Norfolk Hacker YouTube
avatar (hooded hacker with a silver robot face plate and a single glowing
purple cyclops eye). The mask follows head position, size, tilt (roll) and
approximate turn (yaw).

Coverage (option B): face plate + headphones. No hood.

## Target environment

- Parrot OS (Debian-based), x86_64, OBS Studio 30.2.3
- Build deps: `libobs-dev`, `libopencv-dev` (OpenCV ≥ 4.5.4 for `cv::FaceDetectorYN`), `cmake`, `build-essential`
- Installed per-user to `~/.config/obs-studio/plugins/norfolk-hacker-mask/`
  (`bin/64bit/norfolk-hacker-mask.so`, `data/…`) — no root needed for install

## Architecture

Four units, each testable on its own:

1. **`pose` (pure C++, no OBS/OpenCV deps)**
   - Input: 5 landmarks (right eye, left eye, nose, right mouth, left mouth)
     plus detector confidence, in normalized frame coordinates.
   - Output: `MaskPose { cx, cy, scale, roll, yaw, visible }`.
     - centre: eye midpoint plus a vertical offset along the face's down direction
     - scale: inter-eye distance × user size factor
     - roll: `atan2` of the eye-to-eye vector
     - yaw: nose horizontal offset from eye midpoint ÷ inter-eye distance, clamped to [-1, 1]
   - `PoseSmoother`: exponential smoothing per field (roll smoothed via
     angle-wrap-safe interpolation); when the face is lost, hold the last pose
     for 0.5 s then fade opacity to 0 over 0.3 s; fades back in on reacquire.

2. **`tracker` (OpenCV)**
   - Wraps `cv::FaceDetectorYN` with the YuNet ONNX model
     (`face_detection_yunet_2023mar.onnx`, ~230 KB, from opencv_zoo, MIT/Apache
     licensed; shipped in `data/`).
   - Runs on a background thread. Takes the latest downscaled BGRA frame
     (~320 px wide), detects, keeps the highest-confidence face (threshold
     0.6), publishes normalized landmarks. Drops stale frames — always works
     on the newest one.

3. **`filter` (OBS)**
   - `obs_source_info` video filter.
   - In `video_render`: renders the parent source into a `gs_texrender` at
     ~320 px width, copies it to a `gs_stagesurf`, and maps it on the *next*
     frame (avoids GPU stall) → hands pixels to the tracker.
   - Reads the latest landmarks → `pose` → `PoseSmoother` → draws the source
     then the mask layers on top with `gs_matrix_*` transforms.
   - Yaw effect: face plate is squashed horizontally by `1 - 0.3·|yaw|` around the
     eye-socket anchor (no horizontal shift); the near headphone moves outward/grows
     slightly, the far one moves in/shrinks and is drawn behind the plate.
   - Cyclops eye glow: a custom `.effect` shader draws a radial glow at the
     eye socket, colour from settings, intensity pulsing
     `0.75 + 0.25·sin(2π·speed·t)`.

4. **Artwork (`data/`)**
   - Original vector art drawn by `art/make_art.py` (pycairo — no SVG
     converter is installed), rendered to transparent PNGs:
     - `faceplate.png` — silver/gunmetal segmented plate (forehead, cheek
       plates, jaw/mouth grille), purple neon edge lines and circuit traces,
       dark central eye socket.
     - `headphone_l.png`, `headphone_r.png` — dark cups with purple neon rings.
   - Each PNG has a documented anchor point (eye socket centre / ear centre)
     in `src/layout.hpp` so placement is data-driven.
   - Sprites are drawn as explicit 4-corner quads computed by a pure
     `layout` function (no reliance on OBS matrix-stack ordering), which
     the offline preview tool reuses.

## Settings (OBS properties)

| Setting | Default |
|---|---|
| Mask size | 1.0 (range 0.5–2.0) |
| Vertical offset | 0.0 (−0.5–0.5, in inter-eye units) |
| Smoothing | 0.6 (0 = raw, 0.95 = very smooth) |
| Eye glow colour | `#B026FF` |
| Pulse speed | 0.8 Hz (0 = steady) |
| Show headphones | on |
| Debug: show tracking points | off |

## Error handling

- Model file missing / OpenCV fails to load it: log a warning via `blog`,
  filter passes video through unmasked; properties show a status line.
- No face: hold-then-fade as above; no log spam.
- Source size 0 or changes: texrender/stagesurf recreated on resize.
- Tracker thread joined cleanly in `destroy`.

## Testing

- **Unit tests** (CTest, no OBS needed) for `pose` and `PoseSmoother`:
  level face → roll 0; tilted eyes → expected roll; nose offset → yaw sign
  and clamp; scale proportional to eye distance; smoothing converges;
  hold/fade timing; angle wrap around ±π.
- **Offline harness** `mask-preview <image>`: runs the tracker on a photo and
  composites the mask with OpenCV using the same `pose` code, writes a PNG
  for visual inspection.
- **Live check**: load in OBS with `/dev/video0`, verify tracking, tilt,
  turn, fade on face loss, settings changes, CPU usage reasonable.

## Out of scope (v1)

Hood, mouth/jaw animation, multiple faces, Windows/macOS builds, GPU
inference.
