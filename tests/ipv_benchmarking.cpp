#include <doctest/doctest.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "ipv_utils.h"

TEST_CASE(
    "benchmark approximateIpv vs exactIpv — 20 iterations, chrono timed") {
  constexpr int k_iters = 20;
  constexpr size_t k_vertices = 5;
  constexpr size_t k_edges = 12;
  constexpr unsigned k_seed = 42;

  std::mt19937 rng(k_seed);
  const Map map = randomMap(k_vertices, k_edges, rng, 0.25);
  const pMatrix priors(k_edges, 0.3f);

  approximateIpv approx(map, 0.3f);
  exactIpv exact(map, priors);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "iter phase      safe  ig_bits  microseconds\n";
  std::cout << "---- ---------- ---- -------- -------------\n";
  float avg_time_a = 0.0f;
  float avg_time_e = 0.0f;

  float avg_ig_a = 0.0f;
  float avg_ig_e = 0.0f;

  for (int i = 0; i < k_iters; ++i) {
    const Path path = randomPath(k_edges, rng, 0.35);

    const auto t_a0 = std::chrono::high_resolution_clock::now();
    const auto [safe_a, ig_a] = approx.informationGain(path);
    const auto t_a1 = std::chrono::high_resolution_clock::now();

    const auto t_e0 = std::chrono::high_resolution_clock::now();
    const auto [safe_e, ig_e] = exact.informationGain(path);
    const auto t_e1 = std::chrono::high_resolution_clock::now();

    CHECK(safe_a == safe_e);

    const auto us_a =
        std::chrono::duration_cast<std::chrono::microseconds>(t_a1 - t_a0)
            .count();
    const auto us_e =
        std::chrono::duration_cast<std::chrono::microseconds>(t_e1 - t_e0)
            .count();

    // time and ig benchmarking
    avg_time_a += us_a;
    avg_time_e += us_e;
    avg_ig_a += ig_a;
    avg_ig_e += ig_e;

    std::cout << std::setw(4) << i << " approximate " << std::setw(4)
              << std::boolalpha << safe_a << std::noboolalpha << ' '
              << std::setw(8) << ig_a << ' ' << std::setw(13) << us_a << '\n';
    std::cout << std::setw(4) << i << " exact       " << std::setw(4)
              << std::boolalpha << safe_e << std::noboolalpha << ' '
              << std::setw(8) << ig_e << ' ' << std::setw(13) << us_e << '\n';
  }


  avg_time_a /= k_iters;
  avg_time_e /= k_iters;
  avg_ig_a /= k_iters;
  avg_ig_e /= k_iters;
  std::cout << "---- ---------- ---- -------- -------------\n";
  std::cout << "avg TIME approximate " << std::setw(4) << avg_time_a << '\n';
  std::cout << "avg TIME exact       " << std::setw(4) << avg_time_e << '\n';
  std::cout << "avg IG approximate " << std::setw(4) << avg_ig_a << '\n';
  std::cout << "avg IG exact       " << std::setw(4) << avg_ig_e << '\n';
  
}
