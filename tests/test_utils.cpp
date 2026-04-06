#include "test_utils.h"
#include "ipv_utils.h"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sys/stat.h>

static void ensureDir(const std::string &path) {
  mkdir(path.c_str(), 0755);
}

void runGraphBenchmark(const std::string &graph_name, double prior,
                       double p_query, int iterations, bool csv) {
  Graph graph(graph_name);
  graph.randomlyAssignHazards(prior);
  const Map &map = graph.getMap();
  const size_t n_edges = map.size();
  const pMatrix priors(n_edges, prior);

  approximateIpv approx(map, prior);
  exactIpv exact(map, priors);

  std::mt19937 rng(42);

  std::ofstream csv_file;
  if (csv) {
    ensureDir("tests/test_csvs");
    csv_file.open("tests/test_csvs/" + graph_name + ".csv");
    csv_file << "iter,phase,safe,ig_bits,microseconds\n";
  }

  std::cout << std::fixed << std::setprecision(6) << std::boolalpha;
  std::cout << "Graph: " << graph_name << "  edges: " << n_edges
            << "  prior: " << prior << "  p_query: " << p_query
            << "  iters: " << iterations << '\n';
  std::cout << "iter phase       safe   ig_bits  microseconds\n";
  std::cout << "---- ----------- ------ -------- -------------\n";

  double avg_time_a = 0.0, avg_time_e = 0.0;
  double avg_ig_a = 0.0, avg_ig_e = 0.0;

  for (int i = 0; i < iterations; ++i) {
    const Path path = graph.randomPath(rng, p_query);

    const auto t_a0 = std::chrono::high_resolution_clock::now();
    const auto [safe_a, ig_a] = approx.informationGain(path);
    const auto t_a1 = std::chrono::high_resolution_clock::now();

    const auto t_e0 = std::chrono::high_resolution_clock::now();
    const auto [safe_e, ig_e] = exact.informationGain(path);
    const auto t_e1 = std::chrono::high_resolution_clock::now();

    const auto us_a =
        std::chrono::duration_cast<std::chrono::microseconds>(t_a1 - t_a0)
            .count();
    const auto us_e =
        std::chrono::duration_cast<std::chrono::microseconds>(t_e1 - t_e0)
            .count();

    avg_time_a += us_a;
    avg_time_e += us_e;
    avg_ig_a += ig_a;
    avg_ig_e += ig_e;

    std::cout << std::setw(4) << i << " approximate " << std::setw(6) << safe_a
              << ' ' << std::setw(8) << ig_a << ' ' << std::setw(13) << us_a
              << '\n';
    std::cout << std::setw(4) << i << " exact       " << std::setw(6) << safe_e
              << ' ' << std::setw(8) << ig_e << ' ' << std::setw(13) << us_e
              << '\n';

    if (csv) {
      csv_file << i << ",approximate," << safe_a << ',' << ig_a << ',' << us_a
               << '\n';
      csv_file << i << ",exact," << safe_e << ',' << ig_e << ',' << us_e
               << '\n';
    }
  }

  avg_time_a /= iterations;
  avg_time_e /= iterations;
  avg_ig_a /= iterations;
  avg_ig_e /= iterations;
  std::cout << "---- ----------- ------ -------- -------------\n";
  std::cout << "avg  approximate        " << std::setw(8) << avg_ig_a << ' '
            << std::setw(13) << avg_time_a << '\n';
  std::cout << "avg  exact              " << std::setw(8) << avg_ig_e << ' '
            << std::setw(13) << avg_time_e << '\n';

  if (csv) {
    std::cout << "CSV written to tests/test_csvs/" << graph_name << ".csv\n";
  }
}
