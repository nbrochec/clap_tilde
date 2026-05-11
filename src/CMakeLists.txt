
add_library(clap INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/utility.h
        ${CMAKE_CURRENT_SOURCE_DIR}/circular_buffer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/energy_threshold.h
        ${CMAKE_CURRENT_SOURCE_DIR}/leaky_integrator.h
        ${CMAKE_CURRENT_SOURCE_DIR}/bpe_tokenizer.h
        ${CMAKE_CURRENT_SOURCE_DIR}/clap_model.h
        ${CMAKE_CURRENT_SOURCE_DIR}/clap_model_onnx.h
        ${CMAKE_CURRENT_SOURCE_DIR}/clap_classifier.h
)

target_include_directories(clap INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${ORT_INCLUDE_DIR})

target_link_libraries(clap
        INTERFACE
        r8brain
        "${TORCH_LIBRARIES}"
        "${ORT_LIB}")
