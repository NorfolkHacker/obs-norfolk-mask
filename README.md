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

## Artwork

`python3 art/make_art.py data/` regenerates the PNGs. Anchors must match `src/layout.hpp`.

## Preview without OBS

```bash
./build/mask-preview data/models/face_detection_yunet_2023mar.onnx data photo.jpg out.png
```

## Credits

Face detection: YuNet (`face_detection_yunet_2023mar.onnx`) from
[opencv_zoo](https://github.com/opencv/opencv_zoo), MIT licence.
Test photo `tests/data/messi5.jpg` from OpenCV sample data.
