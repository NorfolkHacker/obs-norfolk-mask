# Norfolk Hacker Background — OBS person-segmentation filter

Date: 2026-11-12
Status: design approved in chat (approach 1, scene A), awaiting spec review

## Goal

A second video filter in the existing `norfolk-hacker-mask` plugin,
**"Norfolk Hacker Background"**, that cuts the person out of the webcam image
and replaces everything else with an animated, original purple-neon synthwave
Norfolk scene, with a purple neon glow around the person's outline.

The user's experience of the third-party obs-backgroundremoval plugin was
poor; this filter must stand on its own.

## Usage

Add both filters to the webcam source, **Background above Mask** in the
filter list: the background is replaced first, then the face mask is drawn on
the composited image (face detection still works on it).

## Decisions (from the spike, 2026-11-12)

- Model: **PP-HumanSeg** `human_segmentation_pphumanseg_2023mar.onnx`
  (6163938 bytes) from opencv_zoo, Apache-2.0, loaded with OpenCV DNN 4.10.
  - Input: 1×3×192×192 RGB float, normalised `(x/255 − 0.5)/0.5`, NCHW
    (frame squashed to 192×192, aspect not preserved).
  - Output: 1×2×192×192, already softmaxed (c0+c1 = 1 within 1e-7);
    **channel 1 = person probability**.
  - ≈ 5 ms/frame on this machine (20 threads).
- MediaPipe selfie (2 ms) was rejected: visibly leaked background objects.

## Architecture

Shared building blocks (refactors of reviewed code, behaviour unchanged):

1. **`FrameWorker`** (`src/frame-worker.*`, OpenCV) — "newest frame wins"
   background thread extracted from `FaceTracker`: `start(job)`,
   `submit(rgba, w, h, linesize, scale)`, joins on destruction, catches
   exceptions thrown by the job (logs nothing, drops the frame) so a bad frame
   can never `std::terminate` OBS. `FaceTracker` is rebuilt on it.
2. **`FrameGrabber`** (`src/frame-grabber.*`, OBS) — the capture/downscale/
   one-frame-late staging code extracted from the mask filter, parameterised
   by staging size. The mask filter is rebuilt on it.

New units:

3. **`MaskSmoother`** (`src/mask-smoother.*`, pure C++) — per-pixel temporal
   EMA over successive segmentation results (snap on first result or size
   change; smoothing 0–0.9), plus `to_u8()` for texture upload.
4. **`PersonSegmenter`** (`src/segmenter.*`, OpenCV) — loads PP-HumanSeg,
   runs on a `FrameWorker`, publishes `SegResult { alpha (192×192 floats),
   w, h, seq }`; synchronous `segment_rgb()` for tests/tools.
5. **`background-filter.cpp`** (OBS) — filter id
   `norfolk_hacker_background_filter`. Each frame: hand last frame's staged
   192×192 pixels to the segmenter; capture source; stage; when a new result
   arrives, smooth it and upload to a 192×192 `GS_R8` dynamic texture; draw a
   full-frame quad with `background.effect`.
6. **`data/effects/background.effect`** — composites camera over the
   procedural scene using the upscaled mask (bilinear) with
   `smoothstep(threshold ± feather)`; outline glow from 8 mask taps around each
   pixel; scene drawn procedurally (no loops).
7. **Artwork** — `art/make_art.py` gains `windmill_body.png` (256×512, hub at
   (128,112)) and `windmill_sails.png` (512×512, hub at centre): an original
   Norfolk tower-mill silhouette (tapered tower, cap, fantail, door/windows)
   with neon purple rim lines; four lattice sails.

## The scene (screen uv, v = 0 at top)

- Sky above horizon (v = 0.62): gradient near-black purple → purple glow.
- Stars: hashed grid cells, ~1.5 % lit, twinkling.
- Galaxy: log-spiral arms around (0.72·aspect, 0.20), slow rotation, purple/pink.
- Floor below horizon: perspective neon grid scrolling toward the viewer,
  fading toward the horizon; glowing horizon line.
- Windmill: body at x = 0.16, base v = 0.64, height 0.36 of frame; sails
  rotate around the hub.
- All motion scales with a Speed setting; each element can be switched off.

## Colour handling

The background filter samples and writes without sRGB conversion (camera
passthrough is an identity in gamma space; scene colours are authored in
gamma space). Same as the mask sprites.

## Settings

| Setting | Default |
|---|---|
| Edge threshold | 0.5 (0.1–0.9) |
| Edge softness | 0.08 (0.0–0.3) |
| Mask smoothing | 0.5 (0–0.9) |
| Neon outline | on |
| Outline colour | `#B026FF` (OBS int `0xFFFF26B0`) |
| Outline width | 2.0 mask texels (0.5–6) |
| Animation speed | 1.0 (0–3) |
| Stars / Galaxy / Windmill / Grid | on |
| Debug: show cut-out | off |

## Error handling

- Model fails to load → camera passes through unchanged; properties show a
  status line; warning in OBS log.
- Effect or art fails to load → warning; passthrough (missing art only hides
  the windmill).
- Model loaded but no result yet → passthrough until the first result.
- No person → the whole frame is scene.
- Source size 0 → skip filter. Worker joined on destroy.

## Testing

- Unit (CTest): `MaskSmoother` (snap, EMA, clamp, size change), `to_u8`
  (rounding, clamping), `FrameWorker` (newest-frame-wins, exception in job
  doesn't kill the thread, clean shutdown), `PersonSegmenter` (bad model fails
  cleanly; `tests/data/face.jpg` (public-domain NASA portrait) → person region > 0.5, corners < 0.2;
  async submit publishes), existing 25 tests still pass after the refactors.
- Offline: `bg-preview <model> <in> <out>` writes the cut-out composited over a
  flat purple with the outline, for eyeballing edge quality.
- Live (user): add both filters in OBS; check edges, outline, scene animation,
  every setting; check OBS log for effect compile errors.

## Out of scope

Hair-level matting models, GPU inference, user-supplied background images,
blur-the-real-background mode.
