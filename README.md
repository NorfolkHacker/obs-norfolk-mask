# Norfolk Hacker Mask (OBS filter)

A face-tracking OBS Studio filter that puts a purple-neon cyclops cyborg face plate
and headphones over your webcam face.

## Build & install (Linux, OBS 30)

```bash
sudo apt install libobs-dev libopencv-dev cmake build-essential
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
ctest --test-dir build --output-on-failure
cmake --install build   # → ~/.config/obs-studio/plugins/norfolk-hacker-mask
```

Restart OBS, then on your camera source: **Filters → + → Norfolk Hacker Mask**.

## Settings

| Setting | What it does |
|---|---|
| Mask size | Scales the whole mask |
| Vertical offset | Moves it up/down along your face |
| Smoothing | Higher = steadier, more lag |
| Eye glow colour / Pulse speed | Cyclops eye light |
| Show headphones | Toggle the headphones |
| Debug: show tracking points | Green dots on the 5 detected points |

## Background filter

Add **Norfolk Hacker Background** to the same camera source, *above* Norfolk
Hacker Mask. It cuts you out with PP-HumanSeg and replaces the room with an
animated synthwave Norfolk scene (stars, galaxy, windmill, neon grid) with a
purple neon outline around you.

| Setting | What it does |
|---|---|
| Edge threshold / Edge softness | Where and how softly the cut-out edge falls |
| Mask smoothing | Higher = steadier edges, more lag on fast moves |
| Neon outline / colour / width | Glow around your outline |
| Animation speed | Scene motion (0 = still) |
| Stars / Galaxy / Windmill / Neon grid | Toggle scene elements |
| Debug: show cut-out | Shows the raw person mask |

## Artwork

`python3 art/make_art.py data/` regenerates the PNGs. Anchors must match `src/layout.hpp`.

## Preview without OBS

```bash
./build/mask-preview data/models/face_detection_yunet_2023mar.onnx data photo.jpg out.png
```

## Credits

Face detection: YuNet (`face_detection_yunet_2023mar.onnx`) from
[opencv_zoo](https://github.com/opencv/opencv_zoo), MIT licence (see `data/models/LICENSE`).
Test photo `tests/data/face.jpg`: official NASA portrait of astronaut Sunita Williams (public domain, US government work), via [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Sunita_Williams.jpg), resized to 480×600.

Person segmentation: PP-HumanSeg (human_segmentation_pphumanseg_2023mar.onnx) from opencv_zoo, Apache-2.0 (see `data/models/LICENSE-pphumanseg`).

## Licence

Copyright (C) 2026 Norfolk Hacker.

This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation; either version 2 of the License, or (at your option) any later
version. See [LICENSE](LICENSE).

The bundled models keep their own licences: YuNet (MIT, `data/models/LICENSE`)
and PP-HumanSeg (Apache-2.0, `data/models/LICENSE-pphumanseg`).
