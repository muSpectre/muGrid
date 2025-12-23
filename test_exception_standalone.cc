/**
 * Standalone test for exception traceback on Windows
 * Compile with: cl /EHsc /std:c++17 /I"src/libmugrid" test_exception_standalone.cc build/src/libmugrid/muGrid.dir/Release/exception.obj dbghelp.lib
 */

#include "src/libmugrid/core/exception.hh"
#include <iostream>

void level3() {
    throw muGrid::RuntimeError("Test error from level 3");
}

void level2() {
    level3();
}

void level1() {
    level2();
}

int main() {
    std::cout << "Testing exception with traceback on Windows..." << std::endl;

    try {
        level1();
    } catch (const muGrid::RuntimeError& e) {
        std::cout << "\nCaught exception:\n" << e.what() << std::endl;
        return 0;
    }

    std::cout << "ERROR: Exception was not thrown!" << std::endl;
    return 1;
}
