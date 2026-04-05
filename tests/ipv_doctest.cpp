#include <cmath>

#include <doctest/doctest.h>

#include "ipv.h"

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
