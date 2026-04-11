#pragma once

#include "ipv.h"

#include <string>

struct BenchmarkResult {
  int iteration;
  bool safe_approx;
  bool safe_exact;
  double ig_approx;
  double ig_exact;
  long us_approx;
  long us_exact;
};

/// Run approximate vs exact IPV side-by-side on a named graph config.
/// Prints a formatted table of per-iteration results and summary averages.
/// When @p csv is true, also writes per-iteration rows to
/// tests/test_csvs/<graph_name>.csv.
void runGraphBenchmark(const std::string &graph_name, double prior,
                       double p_query, int iterations, bool csv = false,
                       bool use_joint_ig = true);
