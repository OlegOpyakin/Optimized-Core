cmake_minimum_required(VERSION 3.14)

project(ETC)

# GoogleTest requires at least C++14
set(CMAKE_CXX_FLAGS_RELEASE "-std=c++20 -g -O3")
set(CMAKE_CXX_FLAGS_DEBUG "-std=c++20 -Wall -g -O0 -fsanitize=address -fsanitize=undefined")

add_library( Lib SHARED src/main.cc )
add_executable( MAIN src/main.cc )
target_include_directories(MAIN PUBLIC include/)
target_include_directories(Lib PUBLIC include/)

target_link_libraries(MAIN Lib)