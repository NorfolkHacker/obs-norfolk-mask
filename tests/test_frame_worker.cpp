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
