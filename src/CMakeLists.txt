project("cocotte_runner")
set (COCOTTE_RUNNER_NAME "cocotte")

add_subdirectory(datasources)
add_subdirectory(cocotte)

add_executable(${PROJECT_NAME} "./cocotte.cpp")
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${COCOTTE_RUNNER_NAME})
target_link_libraries(${PROJECT_NAME} datasources cocotte)

set (FORWARD_COMPARATOR_NAME "forward_comparator")


add_executable(${FORWARD_COMPARATOR_NAME} "./forward_comparator.cpp")
target_link_libraries(${FORWARD_COMPARATOR_NAME} datasources cocotte)
