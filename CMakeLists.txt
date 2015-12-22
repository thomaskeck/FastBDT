cmake_minimum_required (VERSION 2.8.12)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project (FastBDT)
set (FastBDT_VERSION_MAJOR 1)
set (FastBDT_VERSION_MINOR 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11 -Wall -Wextra")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ggdb3 -g -std=c++11 -Wall -Wextra")

configure_file (
  "${PROJECT_SOURCE_DIR}/include/FastBDT.h.in"
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
)

include_directories("include/")

set(FastBDT_SOURCES
  "${PROJECT_SOURCE_DIR}/src/FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/IO.cxx"
)

set(FastBDT_HEADERS
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
  "${PROJECT_SOURCE_DIR}/include/IO.h"
)

add_library(FastBDT STATIC ${FastBDT_SOURCES} ${FastBDT_HEADERS})

add_executable(FastBDTMain "${PROJECT_SOURCE_DIR}/src/main.cxx" ${FastBDT_HEADERS})
target_link_libraries(FastBDTMain FastBDT)

add_library(FastBDT_CInterface SHARED "${PROJECT_SOURCE_DIR}/src/c_interface.cxx" ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_CInterface)

find_package(GTest)
if(GTEST_FOUND)
  add_executable(unittests "${PROJECT_SOURCE_DIR}/src/tests.cxx" ${FastBDT_HEADERS})
  target_link_libraries(unittests ${GTEST_BOTH_LIBRARIES} FastBDT pthread)
  message(STATUS  ${GTEST_INCLUDE_DIRS})
  target_include_directories(unittests PUBLIC ${GTEST_INCLUDE_DIRS})
else()
  message(STATUS "Could not find gtest installation, skip building unittests.")
endif()


#find_package(PythonLibs)
#if(PYTHONLIBS_FOUND)
#  add_library(PythonFastBDT SHARED "${PROJECT_SOURCE_DIR}/src/pyinterface.cxx" ${FastBDT_SOURCES} ${FastBDT_HEADERS})
#  target_link_libraries(PythonFastBDT pthread ${PYTHON_LIBRARIES})
#  target_include_directories(PythonFastBDT PUBLIC ${PYTHON_INCLUDE_DIRS})
#else()
#  message(STATUS "Could not find python library installation, skip building python interface.")
#endif()

find_package(ROOT)
if(ROOT_FOUND)
  separate_arguments(ROOT_CXX_FLAGS)
  if (ROOT_VERSION_MAJOR GREATER 5)
    ROOT_GENERATE_DICTIONARY(TMVA_FastBDT_Dict ${FastBDT_HEADERS} "${PROJECT_SOURCE_DIR}/include/LinkDef.h" "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h")
    add_library(TMVAFastBDT SHARED ${FastBDT_SOURCES} "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" "${PROJECT_SOURCE_DIR}/src/MethodFastBDT.cxx" ${FastBDT_HEADERS})
    target_compile_options(TMVAFastBDT PUBLIC ${ROOT_CXX_FLAGS})
    target_include_directories(TMVAFastBDT PUBLIC ${ROOT_INCLUDE_DIR}) 
    target_link_libraries(TMVAFastBDT ${ROOT_LIBRARIES})
  else()
    message(STATUS "Found ROOT installation " ${ROOT_VERSION_MAJOR})
    message(STATUS "Could not find ROOT 6 installation, skip building TMVA interface.")
  endif()
else()
  message(STATUS "Could not find ROOT 6 installation, skip building TMVA interface.")
endif()
