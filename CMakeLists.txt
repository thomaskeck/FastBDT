cmake_minimum_required (VERSION 2.8.12)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project (FastBDT)
set (FastBDT_VERSION_MAJOR 5)
set (FastBDT_VERSION_MINOR 0)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11 -Wall -Wextra -g -march=native")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ggdb3 -g -std=c++11 -Wall -Wextra")

configure_file (
  "${PROJECT_SOURCE_DIR}/include/FastBDT.h.in"
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
)

include_directories("${PROJECT_SOURCE_DIR}/include/" "${PROJECT_BINARY_DIR}/include/")

set(FastBDT_SOURCES
  "${PROJECT_SOURCE_DIR}/src/FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/Classifier.cxx"
  "${PROJECT_SOURCE_DIR}/src/FastBDT_IO.cxx"
)

set(FastBDT_TESTS
  "${PROJECT_SOURCE_DIR}/src/test_all.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_Performance.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_Classifier.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT_IO.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT_C_API.cxx"
)

set(FastBDT_HEADERS
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
  "${PROJECT_BINARY_DIR}/include/Classifier.h"
  "${PROJECT_SOURCE_DIR}/include/FastBDT_IO.h"
)

set(FastBDT_CINTERFACE
    "${PROJECT_SOURCE_DIR}/src/FastBDT_C_API.cxx"
    "${PROJECT_SOURCE_DIR}/include/FastBDT_C_API.h" 
)

add_library(FastBDT_static STATIC ${FastBDT_SOURCES} ${FastBDT_HEADERS})
add_library(FastBDT_CInterface SHARED ${FastBDT_CINTERFACE} ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_CInterface)
add_library(FastBDT_shared SHARED ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_shared)

install(TARGETS FastBDT_static FastBDT_shared FastBDT_CInterface
     LIBRARY DESTINATION lib
      ARCHIVE DESTINATION lib
      RUNTIME DESTINATION bin   
   )

install(FILES ${FastBDT_HEADERS} DESTINATION include)

find_package(GTest)
if(GTEST_FOUND)
    add_executable(unittests ${FastBDT_TESTS} ${FastBDT_HEADERS} ${FastBDT_CINTERFACE})
  target_link_libraries(unittests ${GTEST_BOTH_LIBRARIES} FastBDT_static pthread)
  message(STATUS  ${GTEST_INCLUDE_DIRS})
  target_include_directories(unittests PUBLIC ${GTEST_INCLUDE_DIRS})
  install(TARGETS unittests DESTINATION bin)
else()
  message(STATUS "Could not find gtest installation, skip building unittests.")
endif()

find_program(PYTHON "python3")

if (PYTHON)
    configure_file(
        "${PROJECT_SOURCE_DIR}/setup.py.in"
        "${PROJECT_SOURCE_DIR}/setup.py"
    )

    add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp"
                       COMMAND ${PYTHON} "${CMAKE_CURRENT_BINARY_DIR}/setup.py" build
                       COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_CURRENT_BINARY_DIR}/build/timestamp"
                       DEPENDS ${DEPS})

    add_custom_target(target ALL DEPENDS ${OUTPUT})
    add_custom_command(TARGET FastBDT_shared POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FastBDT_shared> "${PROJECT_SOURCE_DIR}/PyFastBDT/")
    add_custom_command(TARGET FastBDT_CInterface POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:FastBDT_CInterface> "${PROJECT_SOURCE_DIR}/PyFastBDT/")

    install(CODE "execute_process(COMMAND ${PYTHON} ${PROJECT_SOURCE_DIR}/setup.py install --prefix=${CMAKE_INSTALL_PREFIX})")
endif()

set(CPACK_PACKAGE_VERSION "${FastBDT_VERSION_MAJOR}.${FastBDT_VERSION_MINOR}")
set(CPACK_GENERATOR "RPM;DEB;TGZ")
set(CPACK_PACKAGE_NAME "FastBDT")
set(CPACK_PACKAGE_RELEASE 1)
set(CPACK_PACKAGE_CONTACT "thomas.keck2@kit.edu")
set(CPACK_PACKAGE_VENDOR "Private")
set(CPACK_PACKAGING_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX})
set(CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_PACKAGE_RELEASE}.${CMAKE_SYSTEM_PROCESSOR}")

SET(CPACK_DEBIAN_PACKAGE_PRIORITY "optional")
SET(CPACK_DEBIAN_PACKAGE_SECTION "libs")
SET(CPACK_DEBIAN_ARCHITECTURE ${CMAKE_SYSTEM_PROCESSOR})

include(CPack)
