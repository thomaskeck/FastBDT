cmake_minimum_required (VERSION 2.8.12)

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

project (FastBDT)
set (FastBDT_VERSION_MAJOR 3)
set (FastBDT_VERSION_MINOR 4)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -O3 -std=c++11 -Wall -Wextra -g -msse2")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -ggdb3 -g -std=c++11 -Wall -Wextra")

configure_file (
  "${PROJECT_SOURCE_DIR}/include/FastBDT.h.in"
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
)

include_directories("${PROJECT_SOURCE_DIR}/include/")

set(FastBDT_SOURCES
  "${PROJECT_SOURCE_DIR}/src/FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/FastBDT_IO.cxx"
)

set(FastBDT_TESTS
  "${PROJECT_SOURCE_DIR}/src/test_all.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT_IO.cxx"
  "${PROJECT_SOURCE_DIR}/src/test_FastBDT_C_API.cxx"
)

set(FastBDT_HEADERS
  "${PROJECT_BINARY_DIR}/include/FastBDT.h"
  "${PROJECT_SOURCE_DIR}/include/FastBDT_IO.h"
)

set(FastBDT_CINTERFACE
    "${PROJECT_SOURCE_DIR}/src/FastBDT_C_API.cxx"
    "${PROJECT_SOURCE_DIR}/include/FastBDT_C_API.h" 
)

add_library(FastBDT_static STATIC ${FastBDT_SOURCES} ${FastBDT_HEADERS})
add_executable(FastBDTMain "${PROJECT_SOURCE_DIR}/examples/FastBDTMain.cxx" ${FastBDT_HEADERS})
target_link_libraries(FastBDTMain FastBDT_static)
add_library(FastBDT_CInterface SHARED ${FastBDT_CINTERFACE} ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_CInterface)
add_library(FastBDT_shared SHARED ${FastBDT_SOURCES} ${FastBDT_HEADERS})
target_link_libraries(FastBDT_shared)

install(TARGETS FastBDT_static FastBDT_shared FastBDTMain FastBDT_CInterface
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

find_package(ROOT)
if(ROOT_FOUND)
  separate_arguments(ROOT_CXX_FLAGS)
  if (ROOT_VERSION_MAJOR GREATER 5)
    # copy header with dictionary to the build folder so we can use relative include Path
    # this is the only way to have ROOT not store the full path and have a release which is relocate-able
    # also copy inside of the include/root folder to allow to use the headers if they get placed inside 
    # of the root include folder of a release
    file( COPY "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" DESTINATION "${PROJECT_BINARY_DIR}/include/root/" )
    ROOT_GENERATE_DICTIONARY(TMVA_FastBDT_Dict "include/root/MethodFastBDT.h" LINKDEF "${PROJECT_SOURCE_DIR}/include/LinkDef.h")
    add_library(TMVAFastBDT SHARED TMVA_FastBDT_Dict.cxx ${FastBDT_SOURCES} "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" "${PROJECT_SOURCE_DIR}/src/MethodFastBDT.cxx" ${FastBDT_HEADERS})
    target_compile_options(TMVAFastBDT PUBLIC ${ROOT_CXX_FLAGS})
    target_include_directories(TMVAFastBDT PUBLIC ${ROOT_INCLUDE_DIR}) 
    target_link_libraries(TMVAFastBDT ${ROOT_LIBRARIES})
    install(TARGETS TMVAFastBDT DESTINATION ${ROOT_LIBRARY_DIR})
    install(FILES "${CMAKE_BINARY_DIR}/TMVA_FastBDT_Dict_rdict.pcm" DESTINATION ${ROOT_LIBRARY_DIR})
    install(FILES "${PROJECT_SOURCE_DIR}/include/MethodFastBDT.h" ${FastBDT_HEADERS} DESTINATION ${ROOT_INCLUDE_DIR})
  else()
    message(STATUS "Found ROOT installation " ${ROOT_VERSION_MAJOR})
    message(STATUS "Could not find ROOT 6 installation, skip building TMVA interface.")
  endif()
else()
  message(STATUS "Could not find ROOT 6 installation, skip building TMVA interface.")
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
