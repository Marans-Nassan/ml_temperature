# == DO NOT EDIT THE FOLLOWING LINES for the Raspberry Pi Pico VS Code Extension to work ==
if(WIN32)
    set(USERHOME $ENV{USERPROFILE})
else()
    set(USERHOME $ENV{HOME})
endif()
set(sdkVersion 2.2.0)
set(toolchainVersion 14_2_Rel1)
set(picotoolVersion 2.2.0-a4)
set(picoVscode ${USERHOME}/.pico-sdk/cmake/pico-vscode.cmake)
if (EXISTS ${picoVscode})
    include(${picoVscode})
endif()
# ====================================================================================

cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PICO_BOARD pico_w CACHE STRING "Board type")

include(pico_sdk_import.cmake)

project(ml_temperature C CXX ASM)

pico_sdk_init()

# Evita BUILD_TESTING global (alguns projetos puxam tests por padrão)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)

# Importa o pico-tflmicro fora do target ALL (não builda tests automaticamente)
add_subdirectory(pico-tflmicro EXCLUDE_FROM_ALL)

add_executable(${PROJECT_NAME}
    main.c
    lib/ssd1306.c
    lib/aht20.c
)

# main.c usa TFLM (C++). Força compilação como C++.
set_source_files_properties(main.c PROPERTIES LANGUAGE CXX)

pico_set_program_name(${PROJECT_NAME} "ml_temperature")
pico_set_program_version(${PROJECT_NAME} "0.1")

pico_enable_stdio_uart(${PROJECT_NAME} 0)
pico_enable_stdio_usb(${PROJECT_NAME} 1)

target_link_libraries(${PROJECT_NAME}
    pico_stdlib
    pico_multicore
    hardware_i2c
    hardware_pwm
    hardware_watchdog
    pico-tflmicro
)

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_LIST_DIR}/lib
    ${CMAKE_CURRENT_LIST_DIR}/pico-tflmicro
    ${CMAKE_CURRENT_LIST_DIR}/pico-tflmicro/src
)

pico_add_extra_outputs(${PROJECT_NAME})