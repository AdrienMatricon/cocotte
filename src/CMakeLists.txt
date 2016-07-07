project("cocotte-exe")
set (EXECUTABLE_NAME "cocotte")

file(GLOB SOURCE "./*.cpp")

add_subdirectory(datasources)
add_subdirectory(cocotte)
add_subdirectory(simulation)

add_executable(${PROJECT_NAME} ${SOURCE})
set_target_properties(${PROJECT_NAME} PROPERTIES OUTPUT_NAME ${EXECUTABLE_NAME})
target_link_libraries(${PROJECT_NAME} datasources cocotte simulation)
