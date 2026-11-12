cmake_minimum_required(VERSION 3.20)
project(norfolk-hacker-mask VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(NHM_BUILD_PLUGIN "Build the OBS plugin" ON)
include(CTest)

add_library(nhm_core STATIC src/pose.cpp src/smoother.cpp src/layout.cpp src/mask-smoother.cpp)
target_include_directories(nhm_core PUBLIC src)
target_compile_options(nhm_core PRIVATE -Wall -Wextra)

find_package(OpenCV 4.5.4 REQUIRED COMPONENTS core imgproc imgcodecs objdetect dnn)
find_package(Threads REQUIRED)

add_library(nhm_tracker STATIC src/tracker.cpp src/frame-worker.cpp)
target_include_directories(nhm_tracker PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(nhm_tracker PUBLIC nhm_core ${OpenCV_LIBS} Threads::Threads)
target_compile_options(nhm_tracker PRIVATE -Wall -Wextra)

add_executable(mask-preview tools/mask-preview.cpp)
target_link_libraries(mask-preview PRIVATE nhm_tracker)

add_library(nhm_segmenter STATIC src/segmenter.cpp)
target_link_libraries(nhm_segmenter PUBLIC nhm_tracker)
target_compile_options(nhm_segmenter PRIVATE -Wall -Wextra)

add_executable(bg-preview tools/bg-preview.cpp)
target_link_libraries(bg-preview PRIVATE nhm_segmenter)

if(BUILD_TESTING)
  add_executable(test_core tests/test_main.cpp tests/test_pose.cpp tests/test_smoother.cpp tests/test_layout.cpp tests/test_mask_smoother.cpp)
  target_link_libraries(test_core PRIVATE nhm_core)
  add_test(NAME core COMMAND test_core)
  add_executable(test_tracker tests/test_main.cpp tests/test_tracker.cpp)
  target_link_libraries(test_tracker PRIVATE nhm_tracker)
  target_compile_definitions(test_tracker PRIVATE
    NHM_MODEL="${CMAKE_SOURCE_DIR}/data/models/face_detection_yunet_2023mar.onnx"
    NHM_FACE_IMG="${CMAKE_SOURCE_DIR}/tests/data/messi5.jpg")
  add_test(NAME tracker COMMAND test_tracker)
  add_executable(test_frame_worker tests/test_main.cpp tests/test_frame_worker.cpp)
  target_link_libraries(test_frame_worker PRIVATE nhm_tracker)
  add_test(NAME frame_worker COMMAND test_frame_worker)
  add_executable(test_segmenter tests/test_main.cpp tests/test_segmenter.cpp)
  target_link_libraries(test_segmenter PRIVATE nhm_segmenter)
  target_compile_definitions(test_segmenter PRIVATE
    NHM_SEG_MODEL="${CMAKE_SOURCE_DIR}/data/models/human_segmentation_pphumanseg_2023mar.onnx"
    NHM_FACE_IMG="${CMAKE_SOURCE_DIR}/tests/data/messi5.jpg")
  add_test(NAME segmenter COMMAND test_segmenter)
endif()

if(NHM_BUILD_PLUGIN)
  find_package(libobs REQUIRED)
  add_library(norfolk-hacker-mask MODULE src/plugin-main.cpp src/mask-filter.cpp src/frame-grabber.cpp src/obs-image.cpp)
  target_link_libraries(norfolk-hacker-mask PRIVATE OBS::libobs nhm_tracker)
  set_target_properties(norfolk-hacker-mask PROPERTIES PREFIX "")
  target_compile_definitions(norfolk-hacker-mask PRIVATE NHM_VERSION="${PROJECT_VERSION}")

  set(NHM_INSTALL_DIR "$ENV{HOME}/.config/obs-studio/plugins/norfolk-hacker-mask"
      CACHE PATH "Per-user OBS plugin directory")
  install(TARGETS norfolk-hacker-mask LIBRARY DESTINATION "${NHM_INSTALL_DIR}/bin/64bit")
  install(DIRECTORY data/ DESTINATION "${NHM_INSTALL_DIR}/data")
endif()
