#include "shader_compiler.hpp"
//#include <glslang/Include/glslang_c_interface.h>
//#include <glslang/Public/ResourceLimits.h>
#include <shaderc/shaderc.hpp>
#include <stdexcept>
#include <logger.hpp>

namespace util{
namespace shader_compiler{

std::vector<uint32_t> compile(const std::string& code, const robin_hood::unordered_map<std::string, std::string>& defines){
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Like -DMY_DEFINE=1
    for(const auto& [define, val]: defines)
        options.AddMacroDefinition(define, val);
    
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(code, shaderc_compute_shader, "a.comp", options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success)
        throw std::runtime_error{module.GetErrorMessage()};

    return {module.cbegin(), module.cend()};
}
}
}