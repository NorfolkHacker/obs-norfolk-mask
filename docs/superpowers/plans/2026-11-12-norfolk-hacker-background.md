# Norfolk Hacker Background Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a second OBS filter, "Norfolk Hacker Background", to the existing plugin: PP-HumanSeg person segmentation on a worker thread, and a GPU shader that composites the person over an animated purple-neon synthwave Norfolk scene with a neon outline.

**Architecture:** Extract two shared building blocks from reviewed code (`FrameWorker` — newest-frame-wins thread; `FrameGrabber` — OBS capture + one-frame-late staging), rebuild `FaceTracker` and the mask filter on them, then add `MaskSmoother` (pure), `PersonSegmenter` (OpenCV DNN), windmill art, and the background filter + effect.

**Tech Stack:** C++17, CMake, libobs 30.2.3, OpenCV 4.10 DNN, pycairo, CTest mini-harness (`tests/check.hpp`).

**Spec:** `docs/superpowers/specs/2026-11-12-norfolk-hacker-background-design.md` (and the mask spec `docs/superpowers/specs/2026-11-12-norfolk-hacker-mask-design.md` for the existing code).

## Global Constraints

- Repo `/home/norfolkhacker/obs-norfolk-mask`; work on a branch `feat/background-filter` created from the tip of `feat/mask-filter` (stacked; nothing is merged to master without the user's say-so).
- Model: `human_segmentation_pphumanseg_2023mar.onnx`, 6163938 bytes, `https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx`, Apache-2.0. Input 1×3×192×192 RGB, `x/127.5 − 1`, NCHW. Output 1×2×192×192 softmaxed; channel 1 = person.
- Filter id `norfolk_hacker_background_filter`, name `Norfolk Hacker Background`. Log tag `[norfolk-hacker-background]`.
- Scene constants: horizon v = 0.62; windmill x = 0.16, base v = 0.64, height 0.36; galaxy centre (0.72·aspect, 0.20).
- Art: `windmill_body.png` 256×512, hub (128, 112) → hub v-fraction 0.21875; `windmill_sails.png` 512×512, hub at centre, sail tips ≤ 240 px from hub.
- Defaults: threshold 0.5 (0.1–0.9), softness 0.08 (0–0.3), mask smoothing 0.5 (0–0.9), outline on, outline colour `0xFFFF26B0`, outline width 2.0 texels (0.5–6), speed 1.0 (0–3), stars/galaxy/windmill/grid on, debug off.
- No sRGB conversion in the background filter (sample with `gs_effect_set_texture`, framebuffer sRGB off).
- Never use `smoothstep(a, b, x)` with a ≥ b in effects (undefined in GLSL); write `1.0 - smoothstep(b, a, x)`.
- Existing 26+ tests must keep passing after each refactor. Commit trailers: the Co-Authored-By line the agent's own harness specifies.

## File Structure

```
src/frame-worker.hpp/.cpp     NEW  newest-frame-wins worker thread (OpenCV Mat view)
src/tracker.hpp/.cpp          MOD  FaceTracker rebuilt on FrameWorker
src/mask-smoother.hpp/.cpp    NEW  temporal EMA over masks + to_u8 (pure)
src/segmenter.hpp/.cpp        NEW  PersonSegmenter (OpenCV DNN + FrameWorker)
src/frame-grabber.hpp/.cpp    NEW  OBS capture + staging helper
src/obs-image.hpp/.cpp        NEW  load a module PNG as a premultiplied texture
src/mask-filter.cpp           MOD  uses FrameGrabber + obs-image
src/background-filter.cpp     NEW  OBS filter
src/plugin-main.cpp           MOD  registers both filters
data/effects/background.effect NEW
data/models/human_segmentation_pphumanseg_2023mar.onnx NEW (+ LICENSE note)
art/make_art.py               MOD  windmill_body.png, windmill_sails.png
tools/bg-preview.cpp          NEW  offline cut-out preview
tests/test_frame_worker.cpp, tests/test_mask_smoother.cpp, tests/test_segmenter.cpp NEW
```

---

### Task 1: `FrameWorker` + rebuild `FaceTracker` on it

**Files:**
- Create: `src/frame-worker.hpp`, `src/frame-worker.cpp`, `tests/test_frame_worker.cpp`
- Modify: `src/tracker.hpp`, `src/tracker.cpp`, `CMakeLists.txt`

**Interfaces:**
- Produces: `class nhm::FrameWorker { using Job = std::function<void(const cv::Mat &rgba, float scale)>; bool start(Job); bool running() const; void submit(const uint8_t *rgba, int w, int h, int linesize, float scale); void stop(); }` — destructor stops/joins; exceptions thrown by the job are swallowed (frame dropped).
- `FaceTracker` public interface unchanged (`load`, `loaded`, `submit`, `latest`, `detect_bgr`); `load()` is now idempotent (second call returns `loaded()` without touching the detector).

- [ ] **Step 1: Write the failing tests**

`tests/test_frame_worker.cpp`:
```cpp
#include <atomic>
#include <chrono>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#include "check.hpp"
#include "frame-worker.hpp"

using namespace nhm;

template <class Pred>
static bool wait_until(Pred p) {
  for (int i = 0; i < 300; ++i) {
    if (p()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return p();
}

TEST(worker_runs_job_on_submitted_frame) {
  FrameWorker w;
  std::atomic<int> calls{0}, cols{0}, rows{0}, px{0};
  std::atomic<float> scale{0};
  CHECK(w.start([&](const cv::Mat &m, float s) {
    cols = m.cols;
    rows = m.rows;
    px = m.at<cv::Vec4b>(0, 0)[2];
    scale = s;
    ++calls;
  }));
  std::vector<uint8_t> buf(3 * 2 * 4, 7);
  buf[2] = 42;
  w.submit(buf.data(), 3, 2, 12, 1.5f);
  CHECK(wait_until([&] { return calls.load() == 1; }));
  CHECK(cols == 3);
  CHECK(rows == 2);
  CHECK(px == 42);
  CHECK_NEAR(scale.load(), 1.5, 1e-6);
}

TEST(worker_respects_linesize_padding) {
  FrameWorker w;
  std::atomic<int> second_row{-1};
  w.start([&](const cv::Mat &m, float) { second_row = m.at<cv::Vec4b>(1, 0)[0]; });
  std::vector<uint8_t> buf(2 * 16, 0);  // 2 rows, 16-byte stride, 2 px wide
  buf[16] = 99;                         // row 1, pixel 0, byte 0
  w.submit(buf.data(), 2, 2, 16, 1.f);
  CHECK(wait_until([&] { return second_row.load() == 99; }));
}

TEST(worker_survives_throwing_job) {
  FrameWorker w;
  std::atomic<int> calls{0};
  w.start([&](const cv::Mat &, float) {
    if (++calls == 1) throw std::runtime_error("boom");
  });
  std::vector<uint8_t> buf(4, 1);
  w.submit(buf.data(), 1, 1, 4, 1.f);
  CHECK(wait_until([&] { return calls.load() == 1; }));
  w.submit(buf.data(), 1, 1, 4, 1.f);
  CHECK(wait_until([&] { return calls.load() == 2; }));
}

TEST(newest_frame_wins) {
  FrameWorker w;
  std::mutex m;
  std::vector<int> seen;
  std::atomic<bool> release{false};
  std::atomic<int> started{0};
  w.start([&](const cv::Mat &f, float) {
    if (started++ == 0)
      while (!release) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard<std::mutex> l(m);
    seen.push_back(f.at<cv::Vec4b>(0, 0)[0]);
  });
  uint8_t a[4] = {1, 0, 0, 0}, b[4] = {2, 0, 0, 0}, c[4] = {3, 0, 0, 0};
  w.submit(a, 1, 1, 4, 1.f);
  CHECK(wait_until([&] { return started.load() == 1; }));
  w.submit(b, 1, 1, 4, 1.f);
  w.submit(c, 1, 1, 4, 1.f);
  release = true;
  CHECK(wait_until([&] {
    std::lock_guard<std::mutex> l(m);
    return seen.size() == 2;
  }));
  std::this_thread::sleep_for(std::chrono::milliseconds(50));
  std::lock_guard<std::mutex> l(m);
  CHECK(seen.size() == 2);
  CHECK(seen[0] == 1);
  CHECK(seen[1] == 3);
}

TEST(submit_before_start_is_noop_and_stop_is_idempotent) {
  FrameWorker w;
  uint8_t px[4] = {0, 0, 0, 0};
  w.submit(px, 1, 1, 4, 1.f);
  CHECK(!w.running());
  CHECK(w.start([](const cv::Mat &, float) {}));
  CHECK(!w.start([](const cv::Mat &, float) {}));  // already started
  CHECK(w.running());
  w.stop();
  w.stop();
  CHECK(!w.running());
}
```

`CMakeLists.txt`: add `src/frame-worker.cpp` to `nhm_tracker`:
```cmake
add_library(nhm_tracker STATIC src/tracker.cpp src/frame-worker.cpp)
```
and inside `if(BUILD_TESTING)`:
```cmake
  add_executable(test_frame_worker tests/test_main.cpp tests/test_frame_worker.cpp)
  target_link_libraries(test_frame_worker PRIVATE nhm_tracker)
  add_test(NAME frame_worker COMMAND test_frame_worker)
```

- [ ] **Step 2: Run to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: error — `frame-worker.hpp: No such file or directory` (or missing `src/frame-worker.cpp`).

- [ ] **Step 3: Implement `FrameWorker`**

`src/frame-worker.hpp`:
```cpp
#pragma once
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <opencv2/core.hpp>
#include <thread>
#include <vector>

namespace nhm {

// One background thread that always processes the newest submitted RGBA frame.
// Frames submitted while the job is busy replace each other; only the last runs.
class FrameWorker {
 public:
  // rgba is a CV_8UC4 view that is only valid during the call.
  using Job = std::function<void(const cv::Mat &rgba, float scale)>;

  FrameWorker() = default;
  ~FrameWorker();
  FrameWorker(const FrameWorker &) = delete;
  FrameWorker &operator=(const FrameWorker &) = delete;

  // Starts the thread. Returns false if already running or job is empty.
  bool start(Job job);
  bool running() const { return thread_.joinable(); }
  // Copies the frame (honouring linesize). No-op when not running.
  void submit(const uint8_t *rgba, int w, int h, int linesize, float scale);
  // Stops and joins. Safe to call repeatedly.
  void stop();

 private:
  void run();

  Job job_;
  std::thread thread_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::vector<uint8_t> pending_;
  int pw_ = 0, ph_ = 0;
  float pscale_ = 1.f;
  bool has_pending_ = false;
  bool stop_ = false;
};

}  // namespace nhm
```

`src/frame-worker.cpp`:
```cpp
#include "frame-worker.hpp"

#include <cstring>

namespace nhm {

FrameWorker::~FrameWorker() { stop(); }

bool FrameWorker::start(Job job) {
  if (thread_.joinable() || !job) return false;
  job_ = std::move(job);
  {
    std::lock_guard<std::mutex> l(mutex_);
    stop_ = false;
    has_pending_ = false;
  }
  thread_ = std::thread(&FrameWorker::run, this);
  return true;
}

void FrameWorker::stop() {
  {
    std::lock_guard<std::mutex> l(mutex_);
    stop_ = true;
  }
  cv_.notify_all();
  if (thread_.joinable()) thread_.join();
}

void FrameWorker::submit(const uint8_t *rgba, int w, int h, int linesize, float scale) {
  if (!thread_.joinable() || !rgba || w <= 0 || h <= 0 || linesize < w * 4) return;
  {
    std::lock_guard<std::mutex> l(mutex_);
    const size_t row = static_cast<size_t>(w) * 4;
    pending_.resize(row * h);
    for (int y = 0; y < h; ++y)
      std::memcpy(&pending_[row * y], rgba + static_cast<size_t>(y) * linesize, row);
    pw_ = w;
    ph_ = h;
    pscale_ = scale;
    has_pending_ = true;
  }
  cv_.notify_one();
}

void FrameWorker::run() {
  std::vector<uint8_t> buf;
  for (;;) {
    int w, h;
    float s;
    {
      std::unique_lock<std::mutex> l(mutex_);
      cv_.wait(l, [&] { return stop_ || has_pending_; });
      if (stop_) return;
      buf.swap(pending_);
      w = pw_;
      h = ph_;
      s = pscale_;
      has_pending_ = false;
    }
    try {
      job_(cv::Mat(h, w, CV_8UC4, buf.data()), s);
    } catch (...) {
      // A failed frame must never take OBS down; drop it and wait for the next.
    }
  }
}

}  // namespace nhm
```

- [ ] **Step 4: Run FrameWorker tests**

Run: `cmake --build build -j && ./build/test_frame_worker`
Expected: 5 tests `ok`.

- [ ] **Step 5: Rebuild `FaceTracker` on `FrameWorker`**

Replace `src/tracker.hpp` with:
```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#include <string>

#include "frame-worker.hpp"
#include "pose.hpp"

namespace nhm {

struct TrackResult {
  bool has_face = false;
  Landmarks lm{};
  uint64_t seq = 0;  // increments per completed detection; 0 = none yet
};

class FaceTracker {
 public:
  static constexpr float kScoreThreshold = 0.6f;

  FaceTracker() = default;
  FaceTracker(const FaceTracker &) = delete;
  FaceTracker &operator=(const FaceTracker &) = delete;

  // Loads the model and starts the worker. Idempotent: later calls just return loaded().
  bool load(const std::string &model_path);
  bool loaded() const { return loaded_; }

  // Copies the frame; the worker detects on the newest submitted frame.
  void submit(const uint8_t *rgba, int w, int h, int linesize, float scale_to_src);
  TrackResult latest() const;

  // Synchronous detection (also used by the worker).
  TrackResult detect_bgr(const cv::Mat &bgr, float scale_to_src);

 private:
  cv::Ptr<cv::FaceDetectorYN> det_;
  std::mutex det_mutex_;
  std::atomic<bool> loaded_{false};
  std::atomic<uint64_t> seq_{0};

  mutable std::mutex result_mutex_;
  TrackResult result_;

  FrameWorker worker_;  // declared last: destroyed (joined) first
};

}  // namespace nhm
```

Replace `src/tracker.cpp` with:
```cpp
#include "tracker.hpp"

#include <exception>
#include <opencv2/imgproc.hpp>

namespace nhm {

bool FaceTracker::load(const std::string &model_path) {
  if (loaded_) return true;
  cv::Ptr<cv::FaceDetectorYN> det;
  try {
    det = cv::FaceDetectorYN::create(model_path, "", cv::Size(320, 320), kScoreThreshold, 0.3f, 5);
  } catch (const std::exception &) {
    det.release();
  }
  if (det.empty()) return false;
  det_ = det;
  loaded_ = true;
  worker_.start([this](const cv::Mat &rgba, float scale) {
    cv::Mat bgr;
    cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
    TrackResult r = detect_bgr(bgr, scale);
    std::lock_guard<std::mutex> l(result_mutex_);
    result_ = r;
  });
  return true;
}

void FaceTracker::submit(const uint8_t *rgba, int w, int h, int linesize, float scale_to_src) {
  if (loaded_) worker_.submit(rgba, w, h, linesize, scale_to_src);
}

TrackResult FaceTracker::latest() const {
  std::lock_guard<std::mutex> l(result_mutex_);
  return result_;
}

TrackResult FaceTracker::detect_bgr(const cv::Mat &bgr, float scale_to_src) {
  TrackResult r;
  if (!loaded_ || bgr.empty()) return r;
  cv::Mat faces;
  {
    std::lock_guard<std::mutex> l(det_mutex_);
    try {
      det_->setInputSize(bgr.size());
      det_->detect(bgr, faces);
    } catch (const std::exception &) {
      faces.release();
    }
  }
  int best = -1;
  float best_score = kScoreThreshold;
  for (int i = 0; i < faces.rows; ++i) {
    const float sc = faces.at<float>(i, 14);
    if (sc >= best_score) {
      best_score = sc;
      best = i;
    }
  }
  if (best >= 0) {
    r.has_face = true;
    for (int k = 0; k < 5; ++k)
      r.lm.pts[k] = {faces.at<float>(best, 4 + 2 * k) * scale_to_src,
                     faces.at<float>(best, 5 + 2 * k) * scale_to_src};
    r.lm.score = best_score;
  }
  r.seq = ++seq_;
  return r;
}

}  // namespace nhm
```

Add a padded-stride test to `tests/test_tracker.cpp` (GPU staging rows are often padded):
```cpp
TEST(async_submit_with_padded_rows) {
  FaceTracker t;
  CHECK(t.load(NHM_MODEL));
  cv::Mat img = cv::imread(NHM_FACE_IMG), rgba;
  cv::cvtColor(img, rgba, cv::COLOR_BGR2RGBA);
  cv::Mat padded(rgba.rows, rgba.cols + 16, CV_8UC4, cv::Scalar(0, 0, 0, 0));
  rgba.copyTo(padded(cv::Rect(0, 0, rgba.cols, rgba.rows)));
  t.submit(padded.data, rgba.cols, rgba.rows, static_cast<int>(padded.step), 1.f);
  TrackResult r;
  for (int i = 0; i < 200 && r.seq == 0; ++i) {
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    r = t.latest();
  }
  CHECK(r.has_face);
}

TEST(load_is_idempotent) {
  FaceTracker t;
  CHECK(t.load(NHM_MODEL));
  CHECK(t.load("/nonexistent/model.onnx"));  // already loaded: unchanged
  CHECK(t.loaded());
}
```

- [ ] **Step 6: Run the full suite**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: all pass (core, tracker with 6 tests, frame_worker 5), no warnings from our sources.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt src/frame-worker.hpp src/frame-worker.cpp src/tracker.hpp src/tracker.cpp tests/test_frame_worker.cpp tests/test_tracker.cpp
git commit -m "refactor: extract FrameWorker; FaceTracker load is idempotent and worker survives exceptions"
```

---

### Task 2: `MaskSmoother` + `to_u8`

**Files:**
- Create: `src/mask-smoother.hpp`, `src/mask-smoother.cpp`, `tests/test_mask_smoother.cpp`
- Modify: `CMakeLists.txt` (add `src/mask-smoother.cpp` to `nhm_core`; add `tests/test_mask_smoother.cpp` to `test_core`)

**Interfaces:**
- Produces: `class nhm::MaskSmoother { explicit MaskSmoother(float smoothing = 0.5f); void set_smoothing(float); const std::vector<float> &update(const float *fresh, int w, int h); int width() const; int height() const; bool empty() const; }`; `void nhm::to_u8(const std::vector<float> &src, std::vector<uint8_t> &dst)`.

- [ ] **Step 1: Write the failing tests**

`tests/test_mask_smoother.cpp`:
```cpp
#include "check.hpp"
#include "mask-smoother.hpp"

using namespace nhm;

TEST(first_mask_snaps) {
  MaskSmoother s(0.5f);
  CHECK(s.empty());
  const float a[2] = {0.2f, 0.8f};
  const auto &m = s.update(a, 2, 1);
  CHECK(!s.empty());
  CHECK(s.width() == 2 && s.height() == 1);
  CHECK_NEAR(m[0], 0.2, 1e-6);
  CHECK_NEAR(m[1], 0.8, 1e-6);
}

TEST(ema_keeps_smoothing_fraction) {
  MaskSmoother s(0.5f);
  const float zero[2] = {0, 0}, one[2] = {1, 1};
  s.update(zero, 2, 1);
  const auto &m = s.update(one, 2, 1);
  CHECK_NEAR(m[0], 0.5, 1e-6);
  CHECK_NEAR(m[1], 0.5, 1e-6);
}

TEST(zero_smoothing_follows) {
  MaskSmoother s(0.f);
  const float zero[1] = {0}, one[1] = {1};
  s.update(zero, 1, 1);
  CHECK_NEAR(s.update(one, 1, 1)[0], 1.0, 1e-6);
}

TEST(smoothing_clamps_to_0_9) {
  MaskSmoother s(0.5f);
  s.set_smoothing(5.f);
  const float zero[1] = {0}, one[1] = {1};
  s.update(zero, 1, 1);
  CHECK_NEAR(s.update(one, 1, 1)[0], 0.1, 1e-6);
  s.set_smoothing(-1.f);
  CHECK_NEAR(s.update(zero, 1, 1)[0], 0.0, 1e-6);
}

TEST(size_change_snaps) {
  MaskSmoother s(0.9f);
  const float a[1] = {0};
  const float b[4] = {1, 1, 1, 1};
  s.update(a, 1, 1);
  const auto &m = s.update(b, 2, 2);
  CHECK(s.width() == 2 && s.height() == 2);
  CHECK_NEAR(m[3], 1.0, 1e-6);
}

TEST(null_or_bad_input_is_ignored) {
  MaskSmoother s(0.5f);
  const float a[1] = {0.3f};
  s.update(a, 1, 1);
  CHECK_NEAR(s.update(nullptr, 1, 1)[0], 0.3, 1e-6);
  CHECK_NEAR(s.update(a, 0, 1)[0], 0.3, 1e-6);
}

TEST(to_u8_rounds_and_clamps) {
  std::vector<uint8_t> out;
  to_u8({-0.5f, 0.f, 0.5f, 1.f, 2.f}, out);
  CHECK(out.size() == 5);
  CHECK(out[0] == 0);
  CHECK(out[1] == 0);
  CHECK(out[2] == 128);
  CHECK(out[3] == 255);
  CHECK(out[4] == 255);
}
```

Edit `CMakeLists.txt`: `add_library(nhm_core STATIC src/pose.cpp src/smoother.cpp src/layout.cpp src/mask-smoother.cpp)` and append `tests/test_mask_smoother.cpp` to `test_core`'s sources.

- [ ] **Step 2: Run to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: error — `mask-smoother.hpp: No such file or directory`.

- [ ] **Step 3: Implement**

`src/mask-smoother.hpp`:
```cpp
#pragma once
#include <cstdint>
#include <vector>

namespace nhm {

// Per-pixel exponential smoothing across successive segmentation results.
class MaskSmoother {
 public:
  explicit MaskSmoother(float smoothing = 0.5f);
  // Fraction of the previous mask kept per new result, clamped to 0..0.9.
  void set_smoothing(float smoothing);
  // Blends a new w x h result in (snaps on the first result or a size change).
  // Null or empty input leaves the current mask unchanged. Returns the current mask.
  const std::vector<float> &update(const float *fresh, int w, int h);
  int width() const { return w_; }
  int height() const { return h_; }
  bool empty() const { return cur_.empty(); }

 private:
  float smoothing_ = 0.5f;
  int w_ = 0, h_ = 0;
  std::vector<float> cur_;
};

// Clamps to 0..1 and converts to 0..255 with rounding.
void to_u8(const std::vector<float> &src, std::vector<uint8_t> &dst);

}  // namespace nhm
```

`src/mask-smoother.cpp`:
```cpp
#include "mask-smoother.hpp"

#include <algorithm>
#include <cmath>

namespace nhm {

MaskSmoother::MaskSmoother(float smoothing) { set_smoothing(smoothing); }

void MaskSmoother::set_smoothing(float smoothing) { smoothing_ = std::clamp(smoothing, 0.f, 0.9f); }

const std::vector<float> &MaskSmoother::update(const float *fresh, int w, int h) {
  if (!fresh || w <= 0 || h <= 0) return cur_;
  const size_t n = static_cast<size_t>(w) * h;
  if (w != w_ || h != h_ || cur_.size() != n) {
    cur_.assign(fresh, fresh + n);
    w_ = w;
    h_ = h;
    return cur_;
  }
  const float a = 1.f - smoothing_;
  for (size_t i = 0; i < n; ++i) cur_[i] += (fresh[i] - cur_[i]) * a;
  return cur_;
}

void to_u8(const std::vector<float> &src, std::vector<uint8_t> &dst) {
  dst.resize(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    dst[i] = static_cast<uint8_t>(std::lround(std::clamp(src[i], 0.f, 1.f) * 255.f));
}

}  // namespace nhm
```

- [ ] **Step 4: Run tests**

Run: `cmake --build build -j && ctest --test-dir build --output-on-failure`
Expected: all pass (`test_core` gains 7 tests).

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/mask-smoother.hpp src/mask-smoother.cpp tests/test_mask_smoother.cpp
git commit -m "feat: temporal mask smoother and u8 conversion"
```

---

### Task 3: `PersonSegmenter` + model + `bg-preview`

**Files:**
- Create: `src/segmenter.hpp`, `src/segmenter.cpp`, `tests/test_segmenter.cpp`, `tools/bg-preview.cpp`, `data/models/human_segmentation_pphumanseg_2023mar.onnx` (download), `data/models/LICENSE-pphumanseg` (licence note)
- Modify: `CMakeLists.txt`

**Interfaces:**
- Consumes: `FrameWorker` (Task 1).
- Produces: `struct nhm::SegResult { std::vector<float> alpha; int w, h; uint64_t seq; }`; `class nhm::PersonSegmenter { static constexpr int kInputSize = 192; bool load(const std::string&); bool loaded() const; void submit(const uint8_t *rgba, int w, int h, int linesize); SegResult latest() const; bool latest_if_newer(uint64_t after_seq, SegResult &out) const; SegResult segment_rgb(const cv::Mat &rgb); }`.

- [ ] **Step 1: Fetch the model**

```bash
curl -sL -o data/models/human_segmentation_pphumanseg_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/human_segmentation_pphumanseg/human_segmentation_pphumanseg_2023mar.onnx
stat -c %s data/models/human_segmentation_pphumanseg_2023mar.onnx
curl -sL https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/human_segmentation_pphumanseg/LICENSE | head -5
```
Expected size `6163938`. Save the licence text you fetched to `data/models/LICENSE-pphumanseg` (it should be the Apache License 2.0; if that URL 404s, write a short note naming the model, its source URL and "Apache License 2.0" and say so in the report).

- [ ] **Step 2: Write the failing tests + CMake wiring**

`tests/test_segmenter.cpp`:
```cpp
#include <chrono>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>

#include "check.hpp"
#include "segmenter.hpp"

using namespace nhm;

static cv::Mat load_rgb() {
  cv::Mat bgr = cv::imread(NHM_FACE_IMG), rgb;
  cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
  return rgb;
}

TEST(bad_model_fails_cleanly) {
  PersonSegmenter s;
  CHECK(!s.load("/nonexistent/model.onnx"));
  CHECK(!s.loaded());
  CHECK(s.segment_rgb(load_rgb()).alpha.empty());
}

TEST(segments_person_in_sample_photo) {
  PersonSegmenter s;
  CHECK(s.load(NHM_SEG_MODEL));
  SegResult r = s.segment_rgb(load_rgb());
  CHECK(r.w == 192 && r.h == 192);
  CHECK(r.alpha.size() == 192u * 192u);
  CHECK(r.seq > 0);
  CHECK(r.alpha[96 * 192 + 96] > 0.5f);  // torso near the centre
  double corner = 0;
  for (int y = 0; y < 20; ++y)
    for (int x = 0; x < 20; ++x) corner += r.alpha[y * 192 + x];
  CHECK(corner / 400.0 < 0.2);
}

TEST(async_submit_publishes) {
  PersonSegmenter s;
  CHECK(s.load(NHM_SEG_MODEL));
  cv::Mat rgba;
  cv::cvtColor(load_rgb(), rgba, cv::COLOR_RGB2RGBA);
  s.submit(rgba.data, rgba.cols, rgba.rows, static_cast<int>(rgba.step));
  SegResult r;
  for (int i = 0; i < 300 && !s.latest_if_newer(0, r); ++i)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  CHECK(r.seq > 0);
  CHECK(r.alpha.size() == 192u * 192u);
  SegResult again;
  CHECK(!s.latest_if_newer(r.seq, again));  // nothing newer yet
}
```

`CMakeLists.txt` — after the `mask-preview` lines:
```cmake
add_library(nhm_segmenter STATIC src/segmenter.cpp)
target_link_libraries(nhm_segmenter PUBLIC nhm_tracker)
target_compile_options(nhm_segmenter PRIVATE -Wall -Wextra)

add_executable(bg-preview tools/bg-preview.cpp)
target_link_libraries(bg-preview PRIVATE nhm_segmenter)
```
and inside `if(BUILD_TESTING)`:
```cmake
  add_executable(test_segmenter tests/test_main.cpp tests/test_segmenter.cpp)
  target_link_libraries(test_segmenter PRIVATE nhm_segmenter)
  target_compile_definitions(test_segmenter PRIVATE
    NHM_SEG_MODEL="${CMAKE_SOURCE_DIR}/data/models/human_segmentation_pphumanseg_2023mar.onnx"
    NHM_FACE_IMG="${CMAKE_SOURCE_DIR}/tests/data/messi5.jpg")
  add_test(NAME segmenter COMMAND test_segmenter)
```
(`nhm_tracker` already links OpenCV including `dnn`, and contains `FrameWorker`.)

- [ ] **Step 3: Run to verify it fails**

Run: `cmake -S . -B build && cmake --build build -j`
Expected: error — missing `src/segmenter.cpp` / `segmenter.hpp` / `tools/bg-preview.cpp`.

- [ ] **Step 4: Implement**

`src/segmenter.hpp`:
```cpp
#pragma once
#include <atomic>
#include <cstdint>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>
#include <vector>

#include "frame-worker.hpp"

namespace nhm {

struct SegResult {
  std::vector<float> alpha;  // w*h person probabilities, row 0 = top
  int w = 0, h = 0;
  uint64_t seq = 0;  // 0 = none yet
};

// PP-HumanSeg person segmentation (192x192, OpenCV DNN) on a background thread.
class PersonSegmenter {
 public:
  static constexpr int kInputSize = 192;

  PersonSegmenter() = default;
  PersonSegmenter(const PersonSegmenter &) = delete;
  PersonSegmenter &operator=(const PersonSegmenter &) = delete;

  // Loads the model and starts the worker. Idempotent.
  bool load(const std::string &model_path);
  bool loaded() const { return loaded_; }

  // Copies an RGBA frame of any size; the worker segments the newest one.
  void submit(const uint8_t *rgba, int w, int h, int linesize);
  SegResult latest() const;
  // Copies the latest result into out only if its seq > after_seq.
  bool latest_if_newer(uint64_t after_seq, SegResult &out) const;

  // Synchronous segmentation of an RGB (CV_8UC3) image of any size.
  SegResult segment_rgb(const cv::Mat &rgb);

 private:
  cv::dnn::Net net_;
  std::mutex net_mutex_;
  std::atomic<bool> loaded_{false};
  std::atomic<uint64_t> seq_{0};

  mutable std::mutex result_mutex_;
  SegResult result_;

  FrameWorker worker_;  // declared last: destroyed (joined) first
};

}  // namespace nhm
```

`src/segmenter.cpp`:
```cpp
#include "segmenter.hpp"

#include <exception>
#include <opencv2/imgproc.hpp>

namespace nhm {

bool PersonSegmenter::load(const std::string &model_path) {
  if (loaded_) return true;
  cv::dnn::Net net;
  try {
    net = cv::dnn::readNet(model_path);
  } catch (const std::exception &) {
    return false;
  }
  if (net.empty()) return false;
  net_ = net;
  loaded_ = true;
  worker_.start([this](const cv::Mat &rgba, float) {
    cv::Mat rgb;
    cv::cvtColor(rgba, rgb, cv::COLOR_RGBA2RGB);
    SegResult r = segment_rgb(rgb);
    if (r.alpha.empty()) return;
    std::lock_guard<std::mutex> l(result_mutex_);
    result_ = std::move(r);
  });
  return true;
}

void PersonSegmenter::submit(const uint8_t *rgba, int w, int h, int linesize) {
  if (loaded_) worker_.submit(rgba, w, h, linesize, 1.f);
}

SegResult PersonSegmenter::latest() const {
  std::lock_guard<std::mutex> l(result_mutex_);
  return result_;
}

bool PersonSegmenter::latest_if_newer(uint64_t after_seq, SegResult &out) const {
  std::lock_guard<std::mutex> l(result_mutex_);
  if (result_.seq <= after_seq) return false;
  out = result_;
  return true;
}

SegResult PersonSegmenter::segment_rgb(const cv::Mat &rgb) {
  SegResult r;
  if (!loaded_ || rgb.empty() || rgb.type() != CV_8UC3) return r;
  cv::Mat in = rgb;
  if (rgb.cols != kInputSize || rgb.rows != kInputSize)
    cv::resize(rgb, in, cv::Size(kInputSize, kInputSize), 0, 0, cv::INTER_AREA);
  cv::Mat f;
  in.convertTo(f, CV_32FC3, 1.0 / 127.5, -1.0);  // (x/255 - 0.5) / 0.5
  const cv::Mat blob = cv::dnn::blobFromImage(f);  // NCHW, channels already RGB
  cv::Mat out;
  {
    std::lock_guard<std::mutex> l(net_mutex_);
    try {
      net_.setInput(blob);
      out = net_.forward().clone();
    } catch (const std::exception &) {
      out.release();
    }
  }
  if (out.dims != 4 || out.size[1] != 2 || out.size[2] != kInputSize || out.size[3] != kInputSize) return r;
  const float *person = out.ptr<float>(0, 1);  // channel 1 = person (already softmaxed)
  r.alpha.assign(person, person + kInputSize * kInputSize);
  r.w = r.h = kInputSize;
  r.seq = ++seq_;
  return r;
}

}  // namespace nhm
```

`tools/bg-preview.cpp`:
```cpp
// Offline check of the person cut-out: composite the photo over flat dark purple
// with a purple outline band, using the same segmenter as the plugin.
#include <cstdio>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "segmenter.hpp"

int main(int argc, char **argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <model.onnx> <in_image> <out.png>\n", argv[0]);
    return 2;
  }
  cv::Mat img = cv::imread(argv[2]);
  if (img.empty()) {
    std::fprintf(stderr, "cannot read %s\n", argv[2]);
    return 1;
  }
  nhm::PersonSegmenter seg;
  if (!seg.load(argv[1])) {
    std::fprintf(stderr, "cannot load model %s\n", argv[1]);
    return 1;
  }
  cv::Mat rgb;
  cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);
  const nhm::SegResult r = seg.segment_rgb(rgb);
  if (r.alpha.empty()) {
    std::fprintf(stderr, "segmentation failed\n");
    return 1;
  }
  cv::Mat small(r.h, r.w, CV_32F, const_cast<float *>(r.alpha.data())), m;
  cv::resize(small, m, img.size(), 0, 0, cv::INTER_LINEAR);
  // a = smoothstep(0.42, 0.58, m)
  cv::Mat a = cv::min(cv::max((m - 0.42f) / 0.16f, 0.f), 1.f);
  a = a.mul(a).mul(3.f - 2.f * a);
  cv::Mat hard = m > 0.5f, grown, band;
  cv::dilate(hard, grown, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(13, 13)));
  band = grown & ~hard;
  cv::Mat bandf;
  band.convertTo(bandf, CV_32F, 1.0 / 255.0);
  cv::GaussianBlur(bandf, bandf, cv::Size(9, 9), 0);

  cv::Mat out(img.size(), CV_8UC3);
  double coverage = 0;
  for (int y = 0; y < img.rows; ++y)
    for (int x = 0; x < img.cols; ++x) {
      const float av = a.at<float>(y, x), bv = bandf.at<float>(y, x);
      coverage += av;
      const cv::Vec3b c = img.at<cv::Vec3b>(y, x);
      const cv::Vec3f bg(26, 8, 40), glow(255, 38, 176);  // BGR
      cv::Vec3f o;
      for (int k = 0; k < 3; ++k) o[k] = bg[k] * (1 - av) + c[k] * av + glow[k] * bv * (1 - av);
      out.at<cv::Vec3b>(y, x) = cv::Vec3b(cv::saturate_cast<uchar>(o[0]), cv::saturate_cast<uchar>(o[1]),
                                          cv::saturate_cast<uchar>(o[2]));
    }
  std::printf("person coverage %.1f%%\n", 100.0 * coverage / (img.rows * img.cols));
  return cv::imwrite(argv[3], out) ? 0 : 1;
}
```

- [ ] **Step 5: Run tests and the preview**

```bash
cmake --build build -j && ctest --test-dir build --output-on-failure
./build/bg-preview data/models/human_segmentation_pphumanseg_2023mar.onnx tests/data/messi5.jpg /tmp/claude-1000/-home-norfolkhacker/ee3df402-a1fe-4a9f-bd5b-14a3fc51beff/scratchpad/bg_preview.png
```
Expected: all tests pass; preview prints a coverage percentage between 5 and 40; open the PNG with Read — the player is kept, background dark purple, purple glow band around him.

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt src/segmenter.hpp src/segmenter.cpp tests/test_segmenter.cpp tools/bg-preview.cpp data/models/human_segmentation_pphumanseg_2023mar.onnx data/models/LICENSE-pphumanseg
git commit -m "feat: PP-HumanSeg person segmenter with offline cut-out preview"
```

---

### Task 4: Windmill artwork

**Files:**
- Modify: `art/make_art.py`
- Create: `data/windmill_body.png`, `data/windmill_sails.png` (generated)

**Interfaces:**
- Produces: `windmill_body.png` 256×512 RGBA (hub at (128,112), tower base touching y≈500), `windmill_sails.png` 512×512 RGBA (hub at (256,256), tips ≤ 240 px from hub). Consumed by `background.effect` (Task 6).

- [ ] **Step 1: Add the windmill functions**

Add to `art/make_art.py` (above `main`), reusing `rounded_rect`, `neon_stroke`, `PURPLE`, `TAU`:
```python
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
```

In `main()` add:
```python
    make_windmill_body(os.path.join(out, "windmill_body.png"))
    make_windmill_sails(os.path.join(out, "windmill_sails.png"))
```

- [ ] **Step 2: Generate and check**

Run: `python3 art/make_art.py data/ && file data/windmill_*.png && git status --short data/`
Expected: `windmill_body.png: … 256 x 512 … RGBA`, `windmill_sails.png: … 512 x 512 … RGBA`; the existing mask PNGs are byte-identical (git shows them unmodified).
Open both with Read: a dark tower mill silhouette with purple neon edges, cap, fantail, lit windows; four lattice sails in a cross with neon frames. Adjust drawing details if clearly broken; never change canvas sizes or the hub positions.

- [ ] **Step 3: Commit**

```bash
git add art/make_art.py data/windmill_body.png data/windmill_sails.png
git commit -m "feat: Norfolk windmill artwork for the background scene"
```

---

### Task 5: `FrameGrabber` + `obs-image`, rebuild the mask filter on them

**Files:**
- Create: `src/frame-grabber.hpp`, `src/frame-grabber.cpp`, `src/obs-image.hpp`, `src/obs-image.cpp`
- Modify: `src/mask-filter.cpp`, `CMakeLists.txt`

**Interfaces:**
- Produces:
  - `class nhm::FrameGrabber { struct Staged { const uint8_t *rgba; uint32_t width, height, linesize, src_width, src_height; }; template <class Fn> bool take(Fn &&fn); gs_texture_t *capture(obs_source_t *target, uint32_t w, uint32_t h); void stage(gs_texture_t *tex, uint32_t w, uint32_t h, uint32_t sw, uint32_t sh); void release(); }`
  - `bool nhm::load_module_image(gs_image_file3_t *img, const char *module_file, const char *log_tag)` — premultiplied, texture created; logs a warning and returns false on failure.
- Behaviour of the mask filter must be unchanged (including the final-review stale-result guard already in `mf_render`).

- [ ] **Step 1: Write the helpers**

`src/frame-grabber.hpp`:
```cpp
#pragma once
#include <obs-module.h>

#include <cstdint>

namespace nhm {

// Renders an OBS filter's target into a texture and stages a downscaled copy for
// the CPU; the copy is read back one frame later so the GPU never stalls.
// capture/stage/take run inside video_render; release() needs the graphics context.
class FrameGrabber {
 public:
  struct Staged {
    const uint8_t *rgba;
    uint32_t width, height, linesize;
    uint32_t src_width, src_height;  // size of the frame it was downscaled from
  };

  FrameGrabber() = default;
  FrameGrabber(const FrameGrabber &) = delete;
  FrameGrabber &operator=(const FrameGrabber &) = delete;

  // If last frame's copy is ready, maps it, calls fn(const Staged &), unmaps.
  template <class Fn>
  bool take(Fn &&fn) {
    if (!pending_ || !stage_) return false;
    pending_ = false;
    uint8_t *px = nullptr;
    uint32_t linesize = 0;
    if (!gs_stagesurface_map(stage_, &px, &linesize)) return false;
    fn(Staged{px, stage_w_, stage_h_, linesize, src_w_, src_h_});
    gs_stagesurface_unmap(stage_);
    return true;
  }

  // Renders target at w x h. Returns nullptr if rendering failed.
  gs_texture_t *capture(obs_source_t *target, uint32_t w, uint32_t h);
  // Downscales tex (w x h) to sw x sh and queues it for take() on the next frame.
  void stage(gs_texture_t *tex, uint32_t w, uint32_t h, uint32_t sw, uint32_t sh);
  void release();

 private:
  gs_texrender_t *full_ = nullptr;
  gs_texrender_t *small_ = nullptr;
  gs_stagesurf_t *stage_ = nullptr;
  uint32_t stage_w_ = 0, stage_h_ = 0, src_w_ = 0, src_h_ = 0;
  bool pending_ = false;
};

}  // namespace nhm
```

`src/frame-grabber.cpp`:
```cpp
#include "frame-grabber.hpp"

#include <graphics/vec4.h>

namespace nhm {

gs_texture_t *FrameGrabber::capture(obs_source_t *target, uint32_t w, uint32_t h) {
  if (!target || !w || !h) return nullptr;
  if (!full_) full_ = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  if (!full_) return nullptr;
  gs_texrender_reset(full_);
  bool ok = false;
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
  if (gs_texrender_begin(full_, w, h)) {
    struct vec4 clear;
    vec4_zero(&clear);
    gs_clear(GS_CLEAR_COLOR, &clear, 0.f, 0);
    gs_ortho(0.f, static_cast<float>(w), 0.f, static_cast<float>(h), -100.f, 100.f);
    obs_source_video_render(target);
    gs_texrender_end(full_);
    ok = true;
  }
  gs_blend_state_pop();
  return ok ? gs_texrender_get_texture(full_) : nullptr;
}

void FrameGrabber::stage(gs_texture_t *tex, uint32_t w, uint32_t h, uint32_t sw, uint32_t sh) {
  if (!tex || !sw || !sh) return;
  if (!small_) small_ = gs_texrender_create(GS_RGBA, GS_ZS_NONE);
  if (!stage_ || stage_w_ != sw || stage_h_ != sh) {
    gs_stagesurface_destroy(stage_);
    stage_ = gs_stagesurface_create(sw, sh, GS_RGBA);
    stage_w_ = sw;
    stage_h_ = sh;
    pending_ = false;
  }
  if (!small_ || !stage_) return;
  gs_texrender_reset(small_);
  if (!gs_texrender_begin(small_, sw, sh)) return;
  gs_ortho(0.f, static_cast<float>(sw), 0.f, static_cast<float>(sh), -100.f, 100.f);
  gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);
  gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
  while (gs_effect_loop(def, "Draw")) gs_draw_sprite(tex, 0, sw, sh);
  gs_blend_state_pop();
  gs_texrender_end(small_);
  gs_stage_texture(stage_, gs_texrender_get_texture(small_));
  src_w_ = w;
  src_h_ = h;
  pending_ = true;
}

void FrameGrabber::release() {
  gs_texrender_destroy(full_);
  gs_texrender_destroy(small_);
  gs_stagesurface_destroy(stage_);
  full_ = small_ = nullptr;
  stage_ = nullptr;
  pending_ = false;
}

}  // namespace nhm
```

`src/obs-image.hpp`:
```cpp
#pragma once
#include <graphics/image-file.h>

namespace nhm {

// Loads <module data>/<module_file> as a premultiplied-alpha texture.
// Logs a warning with log_tag and returns false on failure. Call outside the
// graphics context (it enters it itself).
bool load_module_image(gs_image_file3_t *img, const char *module_file, const char *log_tag);

}  // namespace nhm
```

`src/obs-image.cpp`:
```cpp
#include "obs-image.hpp"

#include <obs-module.h>

namespace nhm {

bool load_module_image(gs_image_file3_t *img, const char *module_file, const char *log_tag) {
  char *path = obs_module_file(module_file);
  gs_image_file3_init(img, path, GS_IMAGE_ALPHA_PREMULTIPLY);
  obs_enter_graphics();
  gs_image_file3_init_texture(img);
  obs_leave_graphics();
  const bool ok = img->image2.image.loaded && img->image2.image.texture;
  if (!ok) blog(LOG_WARNING, "%s failed to load image %s", log_tag, path ? path : module_file);
  bfree(path);
  return ok;
}

}  // namespace nhm
```

`CMakeLists.txt`: add `src/frame-grabber.cpp src/obs-image.cpp` to the `norfolk-hacker-mask` MODULE sources.

- [ ] **Step 2: Rebuild `mask-filter.cpp` on them**

In `src/mask-filter.cpp`:
1. `#include "frame-grabber.hpp"` and `#include "obs-image.hpp"`.
2. In `struct MaskFilter`, replace the fields `full_tr`, `small_tr`, `stage`, `stage_w`, `stage_h`, `stage_pending`, `stage_to_src` with a single `FrameGrabber grabber;`.
3. Delete the file-local `load_image` function; in `mf_create` call `load_module_image(&f->plate, "faceplate.png", kTag);` (same for `ear_l` / `ear_r`) and remove the two `gs_texrender_create` lines.
4. In `mf_destroy`, replace the two `gs_texrender_destroy` calls and `gs_stagesurface_destroy` with `f->grabber.release();` (still inside `obs_enter_graphics()`).
5. In `mf_render`, replace steps 1–3 (map staged pixels → capture into `full_tr` → downscale/stage) with:
```cpp
  // 1. Hand last frame's staged pixels to the tracker.
  f->grabber.take([&](const FrameGrabber::Staged &s) {
    f->tracker.submit(s.rgba, static_cast<int>(s.width), static_cast<int>(s.height), static_cast<int>(s.linesize),
                      static_cast<float>(s.src_width) / static_cast<float>(s.width));
  });

  // 2. Capture the source at full size.
  gs_texture_t *tex = f->grabber.capture(target, w, h);
  if (!tex) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }

  // 3. Downscale and stage for the tracker.
  gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
  if (f->tracker.loaded()) {
    const uint32_t sw = std::min(kDetectWidth, w);
    const uint32_t sh = std::max(1u, static_cast<uint32_t>(std::lround(static_cast<double>(h) * sw / w)));
    f->grabber.stage(tex, w, h, sw, sh);
  }
```
Keep step 4 (draw camera) and step 5 (pose/stale guard/smoothing/mask) exactly as they are.

- [ ] **Step 3: Build and check**

Run:
```bash
cmake -S . -B build && cmake --build build -j 2>&1 | grep -E "error|warning"; \
ctest --test-dir build --output-on-failure && \
grep -nE "texrender|stagesurf|stage_" src/mask-filter.cpp; \
cmake --install build
```
Expected: no errors/warnings; all tests pass; the grep prints nothing (all capture code lives in `frame-grabber.cpp`); install succeeds.

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt src/frame-grabber.hpp src/frame-grabber.cpp src/obs-image.hpp src/obs-image.cpp src/mask-filter.cpp
git commit -m "refactor: share frame capture/staging and image loading between filters"
```

---

### Task 6: Background filter + effect

**Files:**
- Create: `src/background-filter.cpp`, `data/effects/background.effect`
- Modify: `src/plugin-main.cpp`, `CMakeLists.txt`

**Interfaces:**
- Consumes: `FrameGrabber`, `load_module_image` (Task 5); `PersonSegmenter` (Task 3); `MaskSmoother`, `to_u8` (Task 2); `windmill_*.png` (Task 4).
- Produces: `struct obs_source_info nhm_make_background_filter_info();` registered in `obs_module_load`.

- [ ] **Step 1: Write the effect**

`data/effects/background.effect`:
```hlsl
uniform float4x4 ViewProj;
uniform texture2d image;
uniform texture2d mask;
uniform texture2d mill_body;
uniform texture2d mill_sails;
uniform float elapsed;
uniform float aspect;
uniform float threshold;
uniform float feather;
uniform float4 outline_color;
uniform float outline_on;
uniform float outline_uv;
uniform float show_stars;
uniform float show_galaxy;
uniform float show_mill;
uniform float show_grid;
uniform float debug_mask;

sampler_state lin {
	Filter   = Linear;
	AddressU = Clamp;
	AddressV = Clamp;
};

struct VertInOut {
	float4 pos : POSITION;
	float2 uv  : TEXCOORD0;
};

VertInOut VSDefault(VertInOut vert_in)
{
	VertInOut vert_out;
	vert_out.pos = mul(float4(vert_in.pos.xyz, 1.0), ViewProj);
	vert_out.uv  = vert_in.uv;
	return vert_out;
}

float hash21(float2 p)
{
	return frac(sin(dot(p, float2(127.1, 311.7))) * 43758.5453);
}

float person(float2 uv)
{
	float m = mask.Sample(lin, uv).r;
	return smoothstep(threshold - feather, threshold + feather + 0.0001, m);
}

float inside01(float2 t)
{
	return step(0.0, t.x) * step(t.x, 1.0) * step(0.0, t.y) * step(t.y, 1.0);
}

float3 sky(float2 uv, float2 p)
{
	float h = saturate(uv.y / 0.62);
	float3 col = lerp(float3(0.02, 0.0, 0.05), float3(0.22, 0.04, 0.36), h * h);

	float2 d = p - float2(0.72 * aspect, 0.20);
	float r = length(d) + 0.0001;
	float a = atan2(d.y, d.x);
	float arms = 0.5 + 0.5 * sin(2.0 * a - 9.0 * log(r) + elapsed * 0.15);
	float body = exp(-r * 6.0) * (0.35 + 0.65 * arms * arms);
	float core = exp(-r * 40.0);
	col += show_galaxy * (float3(0.55, 0.20, 0.85) * body + float3(1.0, 0.75, 1.0) * core);

	float2 cell = floor(p * 70.0);
	float2 f = frac(p * 70.0) - 0.5;
	float n = hash21(cell);
	float lit = step(0.985, n);
	float2 off = float2(hash21(cell + 3.1), hash21(cell + 7.7)) - 0.5;
	float sd = length(f - off * 0.6);
	float tw = 0.55 + 0.45 * sin(elapsed * (1.5 + n * 3.0) + n * 60.0);
	col += show_stars * float3(0.9, 0.8, 1.0) * lit * tw * (1.0 - smoothstep(0.0, 0.12, sd));
	return col;
}

float3 ground(float2 uv)
{
	float d = max(uv.y - 0.62, 0.0005);
	float3 col = float3(0.03, 0.0, 0.07);
	float2 g = float2((uv.x - 0.5) * aspect / d, 1.0 / d + elapsed * 0.8);
	float2 f = abs(frac(g * 0.5) - 0.5);
	float wx = min(0.004 * aspect / d, 0.5);
	float wz = min(0.004 / (d * d), 0.5);
	float lx = 1.0 - smoothstep(0.0, wx, f.x);
	float lz = 1.0 - smoothstep(0.0, wz, f.y);
	float lines = max(lx, lz) * smoothstep(0.0, 0.2, d);
	col += show_grid * float3(0.70, 0.18, 1.0) * lines * 1.4;
	return col;
}

float4 windmill(float2 uv)
{
	float hm = 0.36;
	float top = 0.64 - hm;
	float px = (uv.x - 0.16) * aspect;

	float2 bu = float2(px / (hm * 0.5) + 0.5, (uv.y - top) / hm);
	float4 acc = mill_body.Sample(lin, bu) * inside01(bu);

	float2 q = float2(px, uv.y - (top + 0.21875 * hm)) / (0.5 * hm);
	float ang = elapsed * 0.4;
	float c = cos(ang);
	float s = sin(ang);
	float2 su = float2(c * q.x - s * q.y, s * q.x + c * q.y) * 0.5 + 0.5;
	float4 sl = mill_sails.Sample(lin, su) * inside01(su);
	return sl + acc * (1.0 - sl.a);
}

float3 scene(float2 uv)
{
	float2 p = float2(uv.x * aspect, uv.y);
	float3 col = lerp(sky(uv, p), ground(uv), step(0.62, uv.y));
	col += float3(0.9, 0.3, 1.0) * exp(-abs(uv.y - 0.62) * 60.0);
	float4 m = windmill(uv) * show_mill;
	return m.rgb + col * (1.0 - m.a);
}

float halo(float2 uv)
{
	float o = outline_uv;
	float k = o * 0.7071;
	float s = person(uv + float2(o, 0.0));
	s = max(s, person(uv - float2(o, 0.0)));
	s = max(s, person(uv + float2(0.0, o)));
	s = max(s, person(uv - float2(0.0, o)));
	s = max(s, person(uv + float2(k, k)));
	s = max(s, person(uv - float2(k, k)));
	s = max(s, person(uv + float2(k, -k)));
	s = max(s, person(uv - float2(k, -k)));
	return s;
}

float4 PSDraw(VertInOut vert_in) : TARGET
{
	float2 uv = vert_in.uv;
	float a = person(uv);
	float3 cam = image.Sample(lin, uv).rgb;
	float3 col = lerp(scene(uv), cam, a);
	float ring = saturate(halo(uv) - a) * outline_on;
	col += outline_color.rgb * ring * 1.5;
	col = lerp(col, float3(a, a, a), debug_mask);
	return float4(col, 1.0);
}

technique Draw
{
	pass
	{
		vertex_shader = VSDefault(vert_in);
		pixel_shader  = PSDraw(vert_in);
	}
}
```

- [ ] **Step 2: Write the filter**

`src/background-filter.cpp`:
```cpp
#include <graphics/image-file.h>
#include <graphics/vec4.h>
#include <obs-module.h>

#include <cmath>
#include <vector>

#include "frame-grabber.hpp"
#include "mask-smoother.hpp"
#include "obs-image.hpp"
#include "segmenter.hpp"

using namespace nhm;

namespace {

constexpr const char *kTag = "[norfolk-hacker-background]";
constexpr uint32_t kSeg = PersonSegmenter::kInputSize;

struct BackgroundFilter {
  obs_source_t *ctx = nullptr;
  FrameGrabber grabber;
  gs_effect_t *effect = nullptr;
  gs_texture_t *mask_tex = nullptr;
  gs_image_file3_t mill_body{}, mill_sails{};
  bool mill_ok = false;

  PersonSegmenter segmenter;
  MaskSmoother smoother;
  SegResult latest;
  std::vector<uint8_t> mask_u8;
  uint64_t last_seq = 0;

  float threshold = 0.5f, feather = 0.08f, mask_smoothing = 0.5f;
  bool outline = true;
  struct vec4 outline_color {};
  float outline_width = 2.f;
  float speed = 1.f;
  bool stars = true, galaxy = true, windmill = true, grid = true, debug_mask = false;
  double anim_time = 0.0;
};

const char *bf_get_name(void *) { return "Norfolk Hacker Background"; }

void bf_update(void *data, obs_data_t *s) {
  auto *f = static_cast<BackgroundFilter *>(data);
  f->threshold = static_cast<float>(obs_data_get_double(s, "threshold"));
  f->feather = static_cast<float>(obs_data_get_double(s, "feather"));
  f->mask_smoothing = static_cast<float>(obs_data_get_double(s, "mask_smoothing"));
  f->outline = obs_data_get_bool(s, "outline");
  vec4_from_rgba(&f->outline_color, static_cast<uint32_t>(obs_data_get_int(s, "outline_color")));
  f->outline_width = static_cast<float>(obs_data_get_double(s, "outline_width"));
  f->speed = static_cast<float>(obs_data_get_double(s, "speed"));
  f->stars = obs_data_get_bool(s, "stars");
  f->galaxy = obs_data_get_bool(s, "galaxy");
  f->windmill = obs_data_get_bool(s, "windmill");
  f->grid = obs_data_get_bool(s, "grid");
  f->debug_mask = obs_data_get_bool(s, "debug_mask");
}

void bf_defaults(obs_data_t *s) {
  obs_data_set_default_double(s, "threshold", 0.5);
  obs_data_set_default_double(s, "feather", 0.08);
  obs_data_set_default_double(s, "mask_smoothing", 0.5);
  obs_data_set_default_bool(s, "outline", true);
  obs_data_set_default_int(s, "outline_color", 0xFFFF26B0);  // #B026FF (ABGR)
  obs_data_set_default_double(s, "outline_width", 2.0);
  obs_data_set_default_double(s, "speed", 1.0);
  obs_data_set_default_bool(s, "stars", true);
  obs_data_set_default_bool(s, "galaxy", true);
  obs_data_set_default_bool(s, "windmill", true);
  obs_data_set_default_bool(s, "grid", true);
  obs_data_set_default_bool(s, "debug_mask", false);
}

obs_properties_t *bf_properties(void *data) {
  auto *f = static_cast<BackgroundFilter *>(data);
  obs_properties_t *p = obs_properties_create();
  if (f && !f->segmenter.loaded())
    obs_properties_add_text(p, "status", "Segmentation model failed to load: background disabled (see OBS log)",
                            OBS_TEXT_INFO);
  obs_properties_add_float_slider(p, "threshold", "Edge threshold", 0.1, 0.9, 0.01);
  obs_properties_add_float_slider(p, "feather", "Edge softness", 0.0, 0.3, 0.01);
  obs_properties_add_float_slider(p, "mask_smoothing", "Mask smoothing", 0.0, 0.9, 0.01);
  obs_properties_add_bool(p, "outline", "Neon outline");
  obs_properties_add_color(p, "outline_color", "Outline colour");
  obs_properties_add_float_slider(p, "outline_width", "Outline width", 0.5, 6.0, 0.1);
  obs_properties_add_float_slider(p, "speed", "Animation speed", 0.0, 3.0, 0.05);
  obs_properties_add_bool(p, "stars", "Stars");
  obs_properties_add_bool(p, "galaxy", "Galaxy");
  obs_properties_add_bool(p, "windmill", "Windmill");
  obs_properties_add_bool(p, "grid", "Neon grid");
  obs_properties_add_bool(p, "debug_mask", "Debug: show cut-out");
  return p;
}

void *bf_create(obs_data_t *settings, obs_source_t *source) {
  auto *f = new BackgroundFilter();
  f->ctx = source;

  char *model = obs_module_file("models/human_segmentation_pphumanseg_2023mar.onnx");
  if (!model || !f->segmenter.load(model))
    blog(LOG_WARNING, "%s could not load segmentation model (%s); passing video through", kTag,
         model ? model : "missing");
  bfree(model);

  f->mill_ok = load_module_image(&f->mill_body, "windmill_body.png", kTag) &&
               load_module_image(&f->mill_sails, "windmill_sails.png", kTag);

  char *fx = obs_module_file("effects/background.effect");
  obs_enter_graphics();
  char *err = nullptr;
  f->effect = fx ? gs_effect_create_from_file(fx, &err) : nullptr;
  if (!f->effect) blog(LOG_WARNING, "%s failed to load effect: %s", kTag, err ? err : "file not found");
  bfree(err);
  obs_leave_graphics();
  bfree(fx);

  bf_update(f, settings);
  return f;
}

void bf_destroy(void *data) {
  auto *f = static_cast<BackgroundFilter *>(data);
  obs_enter_graphics();
  f->grabber.release();
  gs_texture_destroy(f->mask_tex);
  gs_effect_destroy(f->effect);
  gs_image_file3_free(&f->mill_body);
  gs_image_file3_free(&f->mill_sails);
  obs_leave_graphics();
  delete f;  // joins the segmenter thread
}

void bf_tick(void *data, float seconds) {
  auto *f = static_cast<BackgroundFilter *>(data);
  f->anim_time = std::fmod(f->anim_time + static_cast<double>(seconds) * f->speed, 3600.0);
}

void set_float(gs_effect_t *e, const char *name, float v) {
  gs_effect_set_float(gs_effect_get_param_by_name(e, name), v);
}

void bf_render(void *data, gs_effect_t *) {
  auto *f = static_cast<BackgroundFilter *>(data);
  obs_source_t *target = obs_filter_get_target(f->ctx);
  const uint32_t w = target ? obs_source_get_base_width(target) : 0;
  const uint32_t h = target ? obs_source_get_base_height(target) : 0;
  if (!w || !h || !f->effect || !f->segmenter.loaded()) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }

  // 1. Hand last frame's 192x192 copy to the segmenter.
  f->grabber.take([&](const FrameGrabber::Staged &s) {
    f->segmenter.submit(s.rgba, static_cast<int>(s.width), static_cast<int>(s.height), static_cast<int>(s.linesize));
  });

  // 2. Capture and stage this frame.
  gs_texture_t *tex = f->grabber.capture(target, w, h);
  if (!tex) {
    obs_source_skip_video_filter(f->ctx);
    return;
  }
  f->grabber.stage(tex, w, h, kSeg, kSeg);

  // 3. New segmentation result → smooth → upload.
  if (f->segmenter.latest_if_newer(f->last_seq, f->latest)) {
    f->last_seq = f->latest.seq;
    f->smoother.set_smoothing(f->mask_smoothing);
    to_u8(f->smoother.update(f->latest.alpha.data(), f->latest.w, f->latest.h), f->mask_u8);
    if (!f->mask_tex) f->mask_tex = gs_texture_create(kSeg, kSeg, GS_R8, 1, nullptr, GS_DYNAMIC);
    if (f->mask_tex && f->mask_u8.size() == static_cast<size_t>(kSeg) * kSeg)
      gs_texture_set_image(f->mask_tex, f->mask_u8.data(), kSeg, false);
  }

  const bool prev_srgb = gs_framebuffer_srgb_enabled();
  gs_enable_framebuffer_srgb(false);
  gs_blend_state_push();
  gs_blend_function(GS_BLEND_ONE, GS_BLEND_ZERO);

  if (!f->mask_tex) {
    // No segmentation yet: plain passthrough.
    gs_effect_t *def = obs_get_base_effect(OBS_EFFECT_DEFAULT);
    gs_effect_set_texture(gs_effect_get_param_by_name(def, "image"), tex);
    while (gs_effect_loop(def, "Draw")) gs_draw_sprite(tex, 0, w, h);
  } else {
    gs_effect_t *e = f->effect;
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "image"), tex);
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "mask"), f->mask_tex);
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "mill_body"), f->mill_body.image2.image.texture);
    gs_effect_set_texture(gs_effect_get_param_by_name(e, "mill_sails"), f->mill_sails.image2.image.texture);
    set_float(e, "elapsed", static_cast<float>(f->anim_time));
    set_float(e, "aspect", static_cast<float>(w) / static_cast<float>(h));
    set_float(e, "threshold", f->threshold);
    set_float(e, "feather", f->feather);
    gs_effect_set_vec4(gs_effect_get_param_by_name(e, "outline_color"), &f->outline_color);
    set_float(e, "outline_on", f->outline ? 1.f : 0.f);
    set_float(e, "outline_uv", f->outline_width / static_cast<float>(kSeg));
    set_float(e, "show_stars", f->stars ? 1.f : 0.f);
    set_float(e, "show_galaxy", f->galaxy ? 1.f : 0.f);
    set_float(e, "show_mill", (f->windmill && f->mill_ok) ? 1.f : 0.f);
    set_float(e, "show_grid", f->grid ? 1.f : 0.f);
    set_float(e, "debug_mask", f->debug_mask ? 1.f : 0.f);
    while (gs_effect_loop(e, "Draw")) gs_draw_sprite(tex, 0, w, h);
  }

  gs_blend_state_pop();
  gs_enable_framebuffer_srgb(prev_srgb);
}

}  // namespace

struct obs_source_info nhm_make_background_filter_info() {
  struct obs_source_info info = {};
  info.id = "norfolk_hacker_background_filter";
  info.type = OBS_SOURCE_TYPE_FILTER;
  info.output_flags = OBS_SOURCE_VIDEO;
  info.get_name = bf_get_name;
  info.create = bf_create;
  info.destroy = bf_destroy;
  info.update = bf_update;
  info.get_defaults = bf_defaults;
  info.get_properties = bf_properties;
  info.video_tick = bf_tick;
  info.video_render = bf_render;
  return info;
}
```

- [ ] **Step 3: Register and wire into CMake**

`src/plugin-main.cpp` — declare and register the second filter:
```cpp
struct obs_source_info nhm_make_background_filter_info();
```
and in `obs_module_load` after the mask registration:
```cpp
  struct obs_source_info bg = nhm_make_background_filter_info();
  obs_register_source(&bg);
```

`CMakeLists.txt` — add `src/background-filter.cpp` to the `norfolk-hacker-mask` MODULE sources and link `nhm_segmenter`:
```cmake
  target_link_libraries(norfolk-hacker-mask PRIVATE OBS::libobs nhm_tracker nhm_segmenter)
```

- [ ] **Step 4: Build, test, install**

```bash
cmake -S . -B build && cmake --build build -j 2>&1 | grep -E "error|warning"; \
ctest --test-dir build --output-on-failure && \
nm -D --defined-only build/norfolk-hacker-mask.so | grep obs_module_load && \
cmake --install build && find ~/.config/obs-studio/plugins/norfolk-hacker-mask -type f | sort
```
Expected: no errors or warnings; all tests pass; install lists `effects/background.effect`, `windmill_body.png`, `windmill_sails.png`, `models/human_segmentation_pphumanseg_2023mar.onnx` alongside the existing files.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt src/background-filter.cpp src/plugin-main.cpp data/effects/background.effect
git commit -m "feat: Norfolk Hacker Background filter with synthwave scene and neon outline"
```

---

### Task 7: Live check (user) + README

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Ask the user to restart OBS** and add **Filters → + → Norfolk Hacker Background** on the webcam, placed **above** Norfolk Hacker Mask. Then check the log:
```bash
grep -hE "norfolk-hacker|background.effect|Error|error" "$(ls -t ~/.config/obs-studio/logs/*.txt | head -1)" | tail -40
```
Expected: both filters load; no `failed to load effect` / shader compile errors.

- [ ] **Step 2: User checklist** — person cut out cleanly (hair/shoulders acceptable), outline follows edges, sky/galaxy/stars/windmill/grid animate, each setting works, debug view shows the cut-out, mask filter still tracks on top, CPU acceptable. Fix problems via systematic debugging and repeat.

- [ ] **Step 3: README** — add a "Background filter" section after "Settings":
```markdown
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
```
and add to Credits: `Person segmentation: PP-HumanSeg (human_segmentation_pphumanseg_2023mar.onnx) from opencv_zoo, Apache-2.0.`

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: background filter usage"
```
