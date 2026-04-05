# CMake generated Testfile for 
# Source directory: /home/jmayhugh/repos/matrixstuff
# Build directory: /home/jmayhugh/repos/matrixstuff/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(ipv_doctest "/home/jmayhugh/repos/matrixstuff/build/matrixstuff_tests")
set_tests_properties(ipv_doctest PROPERTIES  _BACKTRACE_TRIPLES "/home/jmayhugh/repos/matrixstuff/CMakeLists.txt;29;add_test;/home/jmayhugh/repos/matrixstuff/CMakeLists.txt;0;")
add_test(ipv_benchmarking "/home/jmayhugh/repos/matrixstuff/build/matrixstuff_benchmarking")
set_tests_properties(ipv_benchmarking PROPERTIES  _BACKTRACE_TRIPLES "/home/jmayhugh/repos/matrixstuff/CMakeLists.txt;31;add_test;/home/jmayhugh/repos/matrixstuff/CMakeLists.txt;0;")
subdirs("_deps/doctest-build")
