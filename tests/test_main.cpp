#include "check.hpp"

std::vector<TestCase> &registry() {
  static std::vector<TestCase> r;
  return r;
}
int g_failures = 0;

int main() {
  for (const auto &t : registry()) {
    int before = g_failures;
    t.fn();
    std::printf("%s %s\n", g_failures == before ? "ok  " : "FAIL", t.name);
  }
  std::printf("%zu tests, %d failed checks\n", registry().size(), g_failures);
  return g_failures == 0 ? 0 : 1;
}
