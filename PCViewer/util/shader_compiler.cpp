#include "shader_compiler.hpp"
#include <stdexcept>
#include <logger.hpp>
#include <sstream>
#include <string_view_util.hpp>
#ifdef _WIN32
#include <shaderc/shaderc.hpp>
#else
#include <glslang/Include/glslang_c_interface.h>
#endif

namespace util{
namespace shader_compiler{
#ifdef _WIN32
std::vector<uint32_t> compile(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines){
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetTargetSpirv(shaderc_spirv_version_1_4);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    // Like -DMY_DEFINE=1
    for(const auto& [define, val]: defines)
        options.AddMacroDefinition(define, val);
    
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(code, shaderc_compute_shader, "a.comp", options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error{module.GetErrorMessage()};

    return {module.cbegin(), module.cend()};
}

std::vector<uint32_t> compile_hlsl(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines, std::string_view entry_point){
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    options.SetTargetSpirv(shaderc_spirv_version_1_4);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);

    // Like -DMY_DEFINE=1
    for(const auto& [define, val]: defines)
        options.AddMacroDefinition(define, val);
    
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult module = compiler.CompileHlslToSpv(code, shaderc_compute_shader, "a.comp", options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error{module.GetErrorMessage()};

    return {module.cbegin(), module.cend()};
}
#else
const glslang_resource_t DefaultTBuiltInResource = {
    /* .MaxLights = */ 32,
    /* .MaxClipPlanes = */ 6,
    /* .MaxTextureUnits = */ 32,
    /* .MaxTextureCoords = */ 32,
    /* .MaxVertexAttribs = */ 64,
    /* .MaxVertexUniformComponents = */ 4096,
    /* .MaxVaryingFloats = */ 64,
    /* .MaxVertexTextureImageUnits = */ 32,
    /* .MaxCombinedTextureImageUnits = */ 80,
    /* .MaxTextureImageUnits = */ 32,
    /* .MaxFragmentUniformComponents = */ 4096,
    /* .MaxDrawBuffers = */ 32,
    /* .MaxVertexUniformVectors = */ 128,
    /* .MaxVaryingVectors = */ 8,
    /* .MaxFragmentUniformVectors = */ 16,
    /* .MaxVertexOutputVectors = */ 16,
    /* .MaxFragmentInputVectors = */ 15,
    /* .MinProgramTexelOffset = */ -8,
    /* .MaxProgramTexelOffset = */ 7,
    /* .MaxClipDistances = */ 8,
    /* .MaxComputeWorkGroupCountX = */ 65535,
    /* .MaxComputeWorkGroupCountY = */ 65535,
    /* .MaxComputeWorkGroupCountZ = */ 65535,
    /* .MaxComputeWorkGroupSizeX = */ 1024,
    /* .MaxComputeWorkGroupSizeY = */ 1024,
    /* .MaxComputeWorkGroupSizeZ = */ 64,
    /* .MaxComputeUniformComponents = */ 1024,
    /* .MaxComputeTextureImageUnits = */ 16,
    /* .MaxComputeImageUniforms = */ 8,
    /* .MaxComputeAtomicCounters = */ 8,
    /* .MaxComputeAtomicCounterBuffers = */ 1,
    /* .MaxVaryingComponents = */ 60,
    /* .MaxVertexOutputComponents = */ 64,
    /* .MaxGeometryInputComponents = */ 64,
    /* .MaxGeometryOutputComponents = */ 128,
    /* .MaxFragmentInputComponents = */ 128,
    /* .MaxImageUnits = */ 8,
    /* .MaxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .MaxCombinedShaderOutputResources = */ 8,
    /* .MaxImageSamples = */ 0,
    /* .MaxVertexImageUniforms = */ 0,
    /* .MaxTessControlImageUniforms = */ 0,
    /* .MaxTessEvaluationImageUniforms = */ 0,
    /* .MaxGeometryImageUniforms = */ 0,
    /* .MaxFragmentImageUniforms = */ 8,
    /* .MaxCombinedImageUniforms = */ 8,
    /* .MaxGeometryTextureImageUnits = */ 16,
    /* .MaxGeometryOutputVertices = */ 256,
    /* .MaxGeometryTotalOutputComponents = */ 1024,
    /* .MaxGeometryUniformComponents = */ 1024,
    /* .MaxGeometryVaryingComponents = */ 64,
    /* .MaxTessControlInputComponents = */ 128,
    /* .MaxTessControlOutputComponents = */ 128,
    /* .MaxTessControlTextureImageUnits = */ 16,
    /* .MaxTessControlUniformComponents = */ 1024,
    /* .MaxTessControlTotalOutputComponents = */ 4096,
    /* .MaxTessEvaluationInputComponents = */ 128,
    /* .MaxTessEvaluationOutputComponents = */ 128,
    /* .MaxTessEvaluationTextureImageUnits = */ 16,
    /* .MaxTessEvaluationUniformComponents = */ 1024,
    /* .MaxTessPatchComponents = */ 120,
    /* .MaxPatchVertices = */ 32,
    /* .MaxTessGenLevel = */ 64,
    /* .MaxViewports = */ 16,
    /* .MaxVertexAtomicCounters = */ 0,
    /* .MaxTessControlAtomicCounters = */ 0,
    /* .MaxTessEvaluationAtomicCounters = */ 0,
    /* .MaxGeometryAtomicCounters = */ 0,
    /* .MaxFragmentAtomicCounters = */ 8,
    /* .MaxCombinedAtomicCounters = */ 8,
    /* .MaxAtomicCounterBindings = */ 1,
    /* .MaxVertexAtomicCounterBuffers = */ 0,
    /* .MaxTessControlAtomicCounterBuffers = */ 0,
    /* .MaxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .MaxGeometryAtomicCounterBuffers = */ 0,
    /* .MaxFragmentAtomicCounterBuffers = */ 1,
    /* .MaxCombinedAtomicCounterBuffers = */ 1,
    /* .MaxAtomicCounterBufferSize = */ 16384,
    /* .MaxTransformFeedbackBuffers = */ 4,
    /* .MaxTransformFeedbackInterleavedComponents = */ 64,
    /* .MaxCullDistances = */ 8,
    /* .MaxCombinedClipAndCullDistances = */ 8,
    /* .MaxSamples = */ 4,
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
    /* .maxMeshOutputVerticesEXT = */ 256,
    /* .maxMeshOutputPrimitivesEXT = */ 256,
    /* .maxMeshWorkGroupSizeX_EXT = */ 128,
    /* .maxMeshWorkGroupSizeY_EXT = */ 128,
    /* .maxMeshWorkGroupSizeZ_EXT = */ 128,
    /* .maxTaskWorkGroupSizeX_EXT = */ 128,
    /* .maxTaskWorkGroupSizeY_EXT = */ 128,
    /* .maxTaskWorkGroupSizeZ_EXT = */ 128,
    /* .maxMeshViewCountEXT = */ 4,
    /* .maxDualSourceDrawBuffersEXT = */ 1,

    /* .limits = */ {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
}};

inline bool line_empty(std::string_view line){
    for(char c: line)
        if(c != ' ' || c!= '\n')
            return false;
    return true;
}
std::vector<uint32_t> compile_impl(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines, std::string_view entry_point, glslang_source_t source_type){
    std::stringstream code_including_defines;
    auto code_by_lines = code | util::slice('\n');
    auto cur = code_by_lines.begin();
    for(std::string_view line = *cur; cur != code_by_lines.end(); line = *++cur){
        code_including_defines << line << '\n';
        if(line.find("#version") != std::string_view::npos)
            break;
    }
    for(const auto [define, val]: defines)
        code_including_defines << "#define " << define << ' ' << val << '\n';
    code_including_defines << cur.get_rest();
    std::string c = code_including_defines.str();
    code_including_defines.str({});
    
    const glslang_input_t input = {
        .language = source_type,
        .stage = GLSLANG_STAGE_COMPUTE,
        .client = GLSLANG_CLIENT_VULKAN,
        .client_version = GLSLANG_TARGET_VULKAN_1_2,
        .target_language = GLSLANG_TARGET_SPV,
		.target_language_version = GLSLANG_TARGET_SPV_1_4,
		.code = c.data(),
		.default_version = 400,
		.default_profile = GLSLANG_NO_PROFILE,
		.force_default_version_and_profile = false,
		.forward_compatible = false,
		.messages = GLSLANG_MSG_DEFAULT_BIT,
		.resource = &DefaultTBuiltInResource
    };

    glslang_initialize_process();


    glslang_shader_t* shader = glslang_shader_create( &input );

    if ( !glslang_shader_preprocess(shader, &input) )
    {
        throw std::runtime_error{glslang_shader_get_info_log(shader)};
    }

    if ( !glslang_shader_parse(shader, &input) )
    {
        throw std::runtime_error{glslang_shader_get_info_log(shader)};
    }

    glslang_program_t* program = glslang_program_create();
    glslang_program_add_shader( program, shader );

    if (!glslang_program_link(program, GLSLANG_MSG_SPV_RULES_BIT | GLSLANG_MSG_VULKAN_RULES_BIT))
    {
        throw std::runtime_error{glslang_shader_get_info_log(shader)};
    }

    glslang_program_SPIRV_generate( program, input.stage );

    if ( glslang_program_SPIRV_get_messages(program) )
    {
        logger << logging::info_prefix << " " << glslang_program_SPIRV_get_messages(program) << logging::endl;
    }

    glslang_shader_delete( shader );

    std::vector<uint32_t> result(glslang_program_SPIRV_get_ptr(program), glslang_program_SPIRV_get_ptr(program) + glslang_program_SPIRV_get_size(program));

    glslang_program_delete( program );
    
    return result;
}
std::vector<uint32_t> compile(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines){
    return compile_impl(code, defines, "main", GLSLANG_SOURCE_GLSL);
}
std::vector<uint32_t> compile_hlsl(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines, std::string_view entry_point){
    return compile_impl(code, defines, entry_point, GLSLANG_SOURCE_HLSL);
}
#endif
}
}