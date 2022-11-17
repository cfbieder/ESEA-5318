cmake_minimum_required(VERSION 3.1)
project( BufferImage )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)
find_package(Threads REQUIRED)


add_executable( BufferImage BufferImage.cpp )
target_link_libraries( BufferImage ${OpenCV_LIBS} )
target_link_libraries( BufferImage ${CMAKE_THREAD_LIBS_INIT})
