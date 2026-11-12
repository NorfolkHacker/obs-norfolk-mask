#pragma once
#include <cmath>
#include <cstdio>
#include <vector>

struct TestCase {
  const char *name;
  void (*fn)();
};
std::vector<TestCase> &registry();
extern int g_failures;

#define TEST(name)                                                    \
  static void name();                                                 \
  static const int name##_reg = (registry().push_back({#name, name}), 0); \
  static void name()

#define CHECK(c)                                                      \
  do {                                                                \
    if (!(c)) {                                                       \
      std::printf("  FAIL %s:%d: %s\n", __FILE__, __LINE__, #c);      \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

#define CHECK_NEAR(a, b, eps)                                         \
  do {                                                                \
    double _a = (a), _b = (b);                                        \
    if (std::fabs(_a - _b) > (eps)) {                                 \
      std::printf("  FAIL %s:%d: %s=%g expected %s=%g\n", __FILE__,   \
                  __LINE__, #a, _a, #b, _b);                          \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)
