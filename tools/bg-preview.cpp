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
