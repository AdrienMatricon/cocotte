project("cocotte_runner")
set (COCOTTE_RUNNER_NAME "cocotte")

add_subdirectory(datasources)
add_subdirectory(cocotte)

add_executable(${PROJECT_NAME} "./cocotte.cpp")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${COCOTTE_RUNNER_NAME})
target_link_libraries(${PROJECT_NAME} datasources cocotte)
