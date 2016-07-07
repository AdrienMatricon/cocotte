project("cocotte")

file(GLOB_RECURSE ${PROJECT_NAME}_HEADER_H "../../include/${PROJECT_NAME}/*.h")
file(GLOB_RECURSE ${PROJECT_NAME}_HEADER_HH "../../include/${PROJECT_NAME}/*.hh")
file(GLOB_RECURSE ${PROJECT_NAME}_HEADER_HPP "../../include/${PROJECT_NAME}/*.hpp")
set(${PROJECT_NAME}_HEADER ${${PROJECT_NAME}_HEADER_H} ${${PROJECT_NAME}_HEADER_HH} ${${PROJECT_NAME}_HEADER_HPP})
file(GLOB_RECURSE ${PROJECT_NAME}_SOURCE "./*.cpp")

find_package(Boost COMPONENTS serialization REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_library(${PROJECT_NAME} ${${PROJECT_NAME}_HEADER} ${${PROJECT_NAME}_SOURCE})

target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} glpk ${CMAKE_BINARY_DIR}/../thirdparty/soplex/lib/libsoplex.a)
