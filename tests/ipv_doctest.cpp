#include <cmath>
#include <limits>

#include <doctest/doctest.h>

#include "ipv.h"
#include "ipv_utils.h"

TEST_CASE("collision when masked edge is obstructed") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m);
  Path path = {false, true};
  CHECK(sim.collision(path));
}

TEST_CASE("safe path when queried edges are clear") {
  Map m = {{0, 0, false}, {0, 1, false}};
  approximateIpv sim(m);
  Path path = {true, true};
  CHECK_FALSE(sim.collision(path));
}

TEST_CASE("informationGain returns safe=false on obstruction") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m);
  Path path = {false, true};
  auto [safe, ig] = sim.informationGain(path);
  CHECK_FALSE(safe);
  CHECK(ig >= 0.f);
}

TEST_CASE("exactIpv five-edge joint update marginals") {
  Map truth = {{0, 1, false},
               {1, 2, false},
               {2, 3, false},
               {3, 4, false},
               {4, 5, false}};
  exactIpv solver(truth, pMatrix(5, 0.2f));
  solver.observe(Path{true, true, true, false, false}, true);
  solver.observe(Path{false, false, true, true, false}, false);
  solver.observe(Path{false, true, false, false, true}, true);
  auto m = solver.marginals();
  REQUIRE(m.size() == 5);
  CHECK(m[0] == doctest::Approx(9.0 / 29.0));
  CHECK(m[1] == doctest::Approx(25.0 / 29.0));
  CHECK(m[2] == doctest::Approx(0.0));
  CHECK(m[3] == doctest::Approx(0.0));
  CHECK(m[4] == doctest::Approx(9.0 / 29.0));
}

TEST_CASE("exactIpv expectedInformationGain is binary entropy of predictive "
          "collision") {
  Map truth = {{0, 0, false}, {0, 1, false}};
  exactIpv solver(truth, pMatrix(2, 0.5f));
  Path q = {true, false};
  const double p = solver.predictiveCollisionProb(q);
  CHECK(solver.expectedInformationGain(q) ==
        doctest::Approx(-p * std::log2(p) - (1.0 - p) * std::log2(1.0 - p)));
}

TEST_CASE("approximateIpv informationGain mutates edge beliefs") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m, 0.5f);
  CHECK(sim.marginals()[0] == doctest::Approx(0.5f));
  CHECK(sim.marginals()[1] == doctest::Approx(0.5f));

  sim.informationGain(Path{false, true});

  CHECK(sim.marginals()[0] == doctest::Approx(0.5f));
  CHECK(sim.marginals()[1] == doctest::Approx(1.0f));
}

TEST_CASE(
    "approximateIpv safe observation zeros beliefs on queried edges only") {
  Map m = {{0, 0, false}, {0, 1, false}};
  approximateIpv sim(m, 0.5f);
  sim.informationGain(Path{true, false});
  CHECK(sim.marginals()[0] == doctest::Approx(0.0f));
  CHECK(sim.marginals()[1] == doctest::Approx(0.5f));
}

TEST_CASE("approximateIpv confirmed-safe edges stay zero after later updates") {
  Map m = {{0, 0, false}, {0, 1, true}};
  approximateIpv sim(m, 0.5f);
  sim.informationGain(Path{true, false});
  CHECK(sim.marginals()[0] == doctest::Approx(0.f));
  sim.informationGain(Path{true, true});
  CHECK(sim.marginals()[0] == doctest::Approx(0.f));
  CHECK(sim.marginals()[1] > 0.f);
}

TEST_CASE("exactIpv informationGain mutates posterior and marginals") {
  Map m = {{0, 0, false}, {0, 1, true}};
  exactIpv sim(m, pMatrix(2, 0.5f));
  const auto m_before = sim.marginals();
  const double p0_before = sim.posterior()[0];

  sim.informationGain(Path{false, true});

  const auto m_after = sim.marginals();
  double sum = 0.0;
  for (double w : sim.posterior()) {
    sum += w;
  }
  CHECK(sum == doctest::Approx(1.0));
  CHECK(m_after[0] == doctest::Approx(m_before[0]));
  CHECK(m_after[1] > m_before[1]);
  CHECK(sim.posterior()[0] != doctest::Approx(p0_before));
}

TEST_CASE("exactIpv observe safe leaves only states with no hazard on queried "
          "edges") {
  Map m = {{0, 1, false}};
  exactIpv sim(m, pMatrix{0.4f});
  sim.observe(Path{true}, false);
  const auto mar = sim.marginals();
  REQUIRE(mar.size() == 1);
  CHECK(mar[0] == doctest::Approx(0.0));
  CHECK(sim.posterior()[0] == doctest::Approx(1.0));
}

TEST_CASE("exactIpv observe collision conditions on hazard along query mask") {
  Map m = {{0, 1, false}, {1, 2, false}};
  exactIpv sim(m, pMatrix{0.5f, 0.5f});
  sim.observe(Path{true, false}, true);
  const auto mar = sim.marginals();
  REQUIRE(mar.size() == 2);
  CHECK(mar[0] == doctest::Approx(1.0));
  CHECK(mar[1] == doctest::Approx(0.5));
}

TEST_CASE("approxiamteIPV and exactIPV produce the same marginals") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.5f);
  exactIpv sim_exact(m, pMatrix{0.5f, 0.5f});

  CHECK(sim.marginals() == sim_exact.marginals());

  // now check after one query
  sim.informationGain(Path{true, false});
  sim_exact.informationGain(Path{true, false});
  CHECK(sim.marginals() == sim_exact.marginals());

  sim.informationGain(Path{true, true});
  sim_exact.informationGain(Path{true, true});
  CHECK(sim.marginals() == sim_exact.marginals());
}

// ====================================================================
// Log-odds internal representation tests
// ====================================================================

TEST_CASE("ipv_utils: probToLogOdds / logOddsToProb round-trip") {
  for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
    const double l = ipv_utils::probToLogOdds(p);
    CHECK(ipv_utils::logOddsToProb(l) == doctest::Approx(p));
  }
}

TEST_CASE("ipv_utils: probToLogOdds boundary values") {
  CHECK(ipv_utils::probToLogOdds(0.0) ==
        -std::numeric_limits<double>::infinity());
  CHECK(ipv_utils::probToLogOdds(1.0) ==
        std::numeric_limits<double>::infinity());
  CHECK(ipv_utils::probToLogOdds(0.5) == doctest::Approx(0.0));
}

TEST_CASE("ipv_utils: softplus basic properties") {
  CHECK(ipv_utils::softplus(0.0) == doctest::Approx(std::log(2.0)));
  CHECK(ipv_utils::softplus(100.0) == doctest::Approx(100.0));
  CHECK(ipv_utils::softplus(-100.0) == doctest::Approx(0.0).epsilon(1e-40));
  for (double x : {-5.0, -1.0, 0.0, 1.0, 5.0}) {
    CHECK(ipv_utils::softplus(x) == doctest::Approx(std::log1p(std::exp(x))));
  }
}

TEST_CASE("ipv_utils: binary_entropy_logodds matches binary_entropy") {
  for (double p : {0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99}) {
    const double l = ipv_utils::probToLogOdds(p);
    CHECK(ipv_utils::binary_entropy_logodds(l) ==
          doctest::Approx(ipv_utils::binary_entropy(p)).epsilon(1e-10));
  }
  CHECK(ipv_utils::binary_entropy_logodds(
            -std::numeric_limits<double>::infinity()) == 0.0);
  CHECK(ipv_utils::binary_entropy_logodds(
            std::numeric_limits<double>::infinity()) == 0.0);
}

TEST_CASE("ipv_utils: total_entropy_logodds matches total_entropy") {
  pMatrix probs = {0.1, 0.5, 0.9, 0.25};
  std::vector<double> lo = ipv_utils::probsToLogOdds(probs);
  CHECK(ipv_utils::total_entropy_logodds(lo) ==
        doctest::Approx(ipv_utils::total_entropy(probs)).epsilon(1e-10));
}

TEST_CASE("ipv_utils: logsumexp correctness") {
  std::vector<double> vals = {-1.0, -2.0, -3.0};
  double expected = std::log(std::exp(-1.0) + std::exp(-2.0) + std::exp(-3.0));
  CHECK(ipv_utils::logsumexp(vals) == doctest::Approx(expected));

  CHECK(ipv_utils::logsumexp({}) ==
        -std::numeric_limits<double>::infinity());
}

TEST_CASE("ipv_utils: log1mexp basic correctness") {
  CHECK(ipv_utils::log1mexp(-1.0) ==
        doctest::Approx(std::log(1.0 - std::exp(-1.0))));
  CHECK(ipv_utils::log1mexp(-0.01) ==
        doctest::Approx(std::log(1.0 - std::exp(-0.01))).epsilon(1e-10));
}

TEST_CASE("approximateIpv internal log-odds: prior 0.5 yields logodds 0") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.5);
  auto probs = sim.marginals();
  for (double p : probs) {
    CHECK(p == doctest::Approx(0.5));
  }
  double lo = ipv_utils::probToLogOdds(0.5);
  CHECK(lo == doctest::Approx(0.0));
}

TEST_CASE("approximateIpv: safe observation drives marginals to zero") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.3);
  sim.observe(Path{true, false}, false);
  auto probs = sim.marginals();
  CHECK(probs[0] == doctest::Approx(0.0));
  CHECK(probs[1] == doctest::Approx(0.3));
}

TEST_CASE("approximateIpv: collision observation increases queried edge "
           "beliefs") {
  Map m = {{0, 1, false}, {1, 2, false}};
  approximateIpv sim(m, 0.3);
  auto probs_before = sim.marginals();
  sim.observe(Path{true, false}, true);
  auto probs_after = sim.marginals();
  CHECK(probs_after[0] > probs_before[0]);
  CHECK(probs_after[1] == doctest::Approx(probs_before[1]));
}

TEST_CASE("exactIpv: log-posterior sums to 1 in probability space") {
  Map m = {{0, 1, false}, {1, 2, false}};
  exactIpv sim(m, pMatrix{0.3, 0.7});

  auto post = sim.posterior();
  double sum = 0.0;
  for (double w : post) {
    sum += w;
  }
  CHECK(sum == doctest::Approx(1.0));

  sim.observe(Path{true, false}, true);
  post = sim.posterior();
  sum = 0.0;
  for (double w : post) {
    sum += w;
  }
  CHECK(sum == doctest::Approx(1.0));
}

TEST_CASE("exactIpv: marginals from log-posterior match hand computation") {
  Map m = {{0, 1, false}};
  exactIpv sim(m, pMatrix{0.4});
  auto mar = sim.marginals();
  REQUIRE(mar.size() == 1);
  CHECK(mar[0] == doctest::Approx(0.4));
}

TEST_CASE("exactIpv: informationGain entropy computed via log-odds path") {
  Map truth = {{0, 1, false}, {1, 2, true}};
  exactIpv sim(truth, pMatrix{0.5, 0.5});

  auto before_probs = sim.marginals();
  auto lo_before = ipv_utils::probsToLogOdds(before_probs);
  double h_before = ipv_utils::total_entropy_logodds(lo_before);

  auto [safe, ig] = sim.informationGain(Path{false, true});

  auto after_probs = sim.marginals();
  auto lo_after = ipv_utils::probsToLogOdds(after_probs);
  double h_after = ipv_utils::total_entropy_logodds(lo_after);

  CHECK(ig == doctest::Approx(h_before - h_after).epsilon(1e-10));
}

TEST_CASE("exactIpv: predictiveCollisionProb matches logsumexp derivation") {
  Map m = {{0, 1, false}, {1, 2, false}};
  exactIpv sim(m, pMatrix{0.3, 0.7});
  double p = sim.predictiveCollisionProb(Path{true, true});
  CHECK(p >= 0.0);
  CHECK(p <= 1.0);
  double expected = 1.0 - (1.0 - 0.3) * (1.0 - 0.7);
  CHECK(p == doctest::Approx(expected));
}
