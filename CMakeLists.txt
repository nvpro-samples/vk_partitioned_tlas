cmake_minimum_required(VERSION 3.22...3.31)
project(vk_partitioned_tlas VERSION 3.0 LANGUAGES C CXX)



# Add the cmake folder to the module path
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
find_package(NvproCore2 REQUIRED)

set(COMMON_DIR "${CMAKE_CURRENT_SOURCE_DIR}/common")


#------------------------------------------------------------------------------------------------------------------------------
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)

# Adding all sources
file(GLOB EXE_SOURCES "src/*.cpp" "src/*.hpp")


source_group("Source Files" FILES ${EXE_SOURCES})

# Add the executable
add_executable(${PROJECT_NAME} ${EXE_SOURCES})



# Link libraries and include directories
target_link_libraries(${PROJECT_NAME} PRIVATE
  nvpro2::nvapp
  nvpro2::nvgui
  nvpro2::nvutils
  nvpro2::nvvk
  nvpro2::nvshaders_host
  nvpro2::nvaftermath
  nvpro2::nvvkglsl
  nvpro2::nvgpu_monitor
)


add_project_definitions(${PROJECT_NAME})

# Include directory for generated files
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_BINARY_DIR} 
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/shaders
)

#------------------------------------------------------------------------------------------------------------------------------
# Compile shaders
file(GLOB SHADER_GLSL_FILES "shaders/*.glsl")
file(GLOB SHADER_H_FILES "shaders/*.h")

set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_LIST_DIR}/_autogen")

compile_glsl(
   "${SHADER_GLSL_FILES}"
   "${SHADER_OUTPUT_DIR}"
   GENERATED_SHADER_GLSL_HEADERS
   EXTRA_FLAGS "-I${NVSHADERS_DIR};-I${COMMON_DIR}"
 )

# Add shader files to the project
source_group("Shaders" FILES ${SHADER_SLANG_FILES} ${SHADER_GLSL_FILES} ${SHADER_H_FILES})
source_group("Shaders/Compiled" FILES ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS})

# Add the output shader headers (target) directly to the executable
# This allows to compile the shaders when the executable is built
target_sources(${PROJECT_NAME} PRIVATE ${GENERATED_SHADER_SLANG_HEADERS} ${GENERATED_SHADER_GLSL_HEADERS} ${SHADER_H_FILES})





#####################################################################################
# Installation, copy files for runtime
copy_to_runtime_and_install(${PROJECT_NAME}
  DIRECTORIES "${NVSHADERS_DIR}/nvshaders"
  FILES ${NsightAftermath_DLLS}
  LOCAL_DIRS "${CMAKE_CURRENT_LIST_DIR}/shaders" 
  AUTO
)



# Make Visual Studio use this project as the startup project
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})

# Install the resources and nvshaders folders
install(DIRECTORY "${NVSHADERS_DIR}/nvshaders" DESTINATION ${CMAKE_INSTALL_PREFIX})


