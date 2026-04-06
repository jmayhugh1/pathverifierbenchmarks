cmake_minimum_required(VERSION 3.14)
project(matrixstuff VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(matrixstuff main.cpp ipv.cpp ipv_utils.cpp)

include(FetchContent)
FetchContent_Declare(
  doctest
  GIT_REPOSITORY https://github.com/doctest/doctest.git
  GIT_TAG v2.5.1
)
FetchContent_MakeAvailable(doctest)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.12.0/json.tar.xz
)

FetchContent_MakeAvailable(json)


add_executable(matrixstuff_tests tests/doctest_main.cpp tests/ipv_doctest.cpp
                              ipv.cpp ipv_utils.cpp)

add_executable(matrixstuff_benchmarking tests/doctest_main.cpp
                                        tests/ipv_benchmarking.cpp ipv.cpp ipv_utils.cpp)
                              
add_executable(matrixstuff_graph_configs tests/doctest_main.cpp tests/ipv_graph_configs.cpp ipv.cpp ipv_utils.cpp)

target_link_libraries(matrixstuff PRIVATE nlohmann_json::nlohmann_json)

target_link_libraries(matrixstuff_tests PRIVATE doctest::doctest nlohmann_json::nlohmann_json)
target_include_directories(matrixstuff_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(matrixstuff_benchmarking PRIVATE doctest::doctest nlohmann_json::nlohmann_json)
target_include_directories(matrixstuff_benchmarking PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(matrixstuff_graph_configs PRIVATE doctest::doctest nlohmann_json::nlohmann_json)
target_include_directories(matrixstuff_graph_configs PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
enable_testing()
add_test(NAME ipv_doctest COMMAND matrixstuff_tests)
add_test(NAME ipv_benchmarking COMMAND matrixstuff_benchmarking)

add_test(NAME ipv_graph_configs COMMAND matrixstuff_graph_configs)
set_tests_properties(ipv_graph_configs PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
