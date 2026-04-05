# CMake generated Testfile for 
# Source directory: /Users/joshuamayhugh/Projects/pathverifierbenchmarks
# Build directory: /Users/joshuamayhugh/Projects/pathverifierbenchmarks/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ipv_doctest "/Users/joshuamayhugh/Projects/pathverifierbenchmarks/build/matrixstuff_tests")
set_tests_properties(ipv_doctest PROPERTIES  _BACKTRACE_TRIPLES "/Users/joshuamayhugh/Projects/pathverifierbenchmarks/CMakeLists.txt;29;add_test;/Users/joshuamayhugh/Projects/pathverifierbenchmarks/CMakeLists.txt;0;")
add_test(ipv_benchmarking "/Users/joshuamayhugh/Projects/pathverifierbenchmarks/build/matrixstuff_benchmarking")
set_tests_properties(ipv_benchmarking PROPERTIES  _BACKTRACE_TRIPLES "/Users/joshuamayhugh/Projects/pathverifierbenchmarks/CMakeLists.txt;31;add_test;/Users/joshuamayhugh/Projects/pathverifierbenchmarks/CMakeLists.txt;0;")
subdirs("_deps/doctest-build")
