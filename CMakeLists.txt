cmake_minimum_required (VERSION 2.8.12)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project (FastBDT)
set (FastBDT_VERSION_MAJOR 1)
set (FastBDT_VERSION_MINOR 2)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11 -Wall -Wextra -g")
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

set(FastBDT_TESTS
  "${PROJECT_SOURCE_DIR}/src/test_all.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_IO.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_c_interface.cxx"
)

set(FastBDT_HEADERS
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
  "${PROJECT_SOURCE_DIR}/include/IO.h"
)

set(FastBDT_CINTERFACE
  "${PROJECT_SOURCE_DIR}/src/c_interface.cxx"
  "${PROJECT_SOURCE_DIR}/include/c_interface.h" 
)

add_library(FastBDT STATIC ${FastBDT_SOURCES} ${FastBDT_HEADERS})
add_executable(FastBDTMain "${PROJECT_SOURCE_DIR}/src/main.cxx" ${FastBDT_HEADERS})
target_link_libraries(FastBDTMain FastBDT)
add_library(FastBDT_CInterface SHARED ${FastBDT_CINTERFACE} ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_CInterface)

install(TARGETS FastBDT FastBDTMain FastBDT_CInterface
   LIBRARY DESTINATION lib
   ARCHIVE DESTINATION lib
   RUNTIME DESTINATION bin   
)

install(FILES ${FastBDT_HEADERS} DESTINATION include)

find_package(GTest)
if(GTEST_FOUND)
    add_executable(unittests ${FastBDT_TESTS} ${FastBDT_HEADERS} ${FastBDT_CINTERFACE})
  target_link_libraries(unittests ${GTEST_BOTH_LIBRARIES} FastBDT pthread)
  message(STATUS  ${GTEST_INCLUDE_DIRS})
  target_include_directories(unittests PUBLIC ${GTEST_INCLUDE_DIRS})
else()
  message(STATUS "Could not find gtest installation, skip building unittests.")
endif()

find_package(ROOT)
if(ROOT_FOUND)
  separate_arguments(ROOT_CXX_FLAGS)
  if (ROOT_VERSION_MAJOR GREATER 5)
    ROOT_GENERATE_DICTIONARY(TMVA_FastBDT_Dict ${FastBDT_HEADERS} "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" LINKDEF "${PROJECT_SOURCE_DIR}/include/LinkDef.h")
    add_library(TMVAFastBDT SHARED TMVA_FastBDT_Dict.cxx ${FastBDT_SOURCES} "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" "${PROJECT_SOURCE_DIR}/src/MethodFastBDT.cxx" ${FastBDT_HEADERS})
    target_compile_options(TMVAFastBDT PUBLIC ${ROOT_CXX_FLAGS})
    target_include_directories(TMVAFastBDT PUBLIC ${ROOT_INCLUDE_DIR}) 
    target_link_libraries(TMVAFastBDT ${ROOT_LIBRARIES})
    install(TARGETS TMVAFastBDT DESTINATION ${ROOT_LIBRARY_DIR})
    install(FILES TMVA_FastBDT_Dict_rdict.pcm DESTINATION ${ROOT_LIBRARY_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" ${FastBDT_HEADERS} DESTINATION ${ROOT_INCLUDE_DIR})
  else()
    message(STATUS "Found ROOT installation " ${ROOT_VERSION_MAJOR})
    message(STATUS "Could not find ROOT 6 installation, skip building TMVA interface.")
  endif()
else()
  message(STATUS "Could not find ROOT 6 installation, skip building TMVA interface.")
endif()
