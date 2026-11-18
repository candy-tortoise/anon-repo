cmake_minimum_required(VERSION 3.16)

project(verification CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

find_package(OpenFHE REQUIRED)

file(GLOB LIB_SOURCES "src/*.cpp")
add_library(${PROJECT_NAME}_lib STATIC ${LIB_SOURCES})

target_compile_options(${PROJECT_NAME}_lib PUBLIC "SHELL:${OpenFHE_CXX_FLAGS}")
target_include_directories(${PROJECT_NAME}_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/third-party/include
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
)
target_link_libraries(${PROJECT_NAME}_lib PUBLIC ${OpenFHE_SHARED_LIBRARIES})

file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
foreach(EXAMPLE_SRC ${EXAMPLE_SOURCES})
    get_filename_component(EXAMPLE_NAME ${EXAMPLE_SRC} NAME_WE)
    add_executable(${EXAMPLE_NAME} ${EXAMPLE_SRC})
    target_link_libraries(${EXAMPLE_NAME} PRIVATE ${PROJECT_NAME}_lib)
endforeach()
