#pragma once

#include <vector>
#include <string>
#include <robin_hood.h>
#include <memory_view.hpp>

namespace structures{
struct commandline_parser{
    struct CommandLineOption {
		std::vector<std::string> commands;
		std::string value;
		bool hasValue = false;
		std::string help;
		bool set = false;
	};
	robin_hood::unordered_map<std::string, CommandLineOption> options;

	commandline_parser(){
        add("help", { "--help" }, false, "Show help");
	    add("fullscreen", { "-f", "--fullscreen" }, false, "Start in fullscreen mode");
	    add("width", { "-w", "--width" }, true, "Set window width");
	    add("height", { "-h", "--height" }, true, "Set window height");
	    add("gpuselection", { "-g", "--gpu" }, true, "Select GPU to run on");
	    add("gpulist", { "-gl", "--listgpus" }, false, "Display a list of available Vulkan devices");
        add("jsonsettings", { "-js", "--jsonsettings" }, true, "Set json settings file. For available json commands see xxx");
    }
	void add(std::string name, std::vector<std::string> commands, bool hasValue, std::string help){
        options[name].commands = commands;
	    options[name].help = help;
	    options[name].set = false;
	    options[name].hasValue = hasValue;
	    options[name].value = "";
    }
	void printHelp(){
        std::cout << "Available command line options:\n";
	    for (auto option : options) {
	    	std::cout << " ";
	    	for (size_t i = 0; i < option.second.commands.size(); i++) {
	    		std::cout << option.second.commands[i];
	    		if (i < option.second.commands.size() - 1) {
	    			std::cout << ", ";
	    		}
	    	}
	    	std::cout << ": " << option.second.help << "\n";
	    }
	    std::cout << "Press any key to close...";
    }
	void parse(util::memory_view<const char*> arguments){
        bool printHelp = false;
	    // Known arguments
	    for (auto& option : options) {
	    	for (auto& command : option.second.commands) {
	    		for (size_t i = 0; i < arguments.size(); i++) {
	    			if (strcmp(arguments[i], command.c_str()) == 0) {
	    				option.second.set = true;
	    				// Get value
	    				if (option.second.hasValue) {
	    					if (arguments.size() > i + 1) {
	    						option.second.value = arguments[i + 1];
	    					}
	    					if (option.second.value == "") {
	    						printHelp = true;
	    						break;
	    					}
	    				}
	    			}
	    		}
	    	}
	    }
	    // Print help for unknown arguments or missing argument values
	    if (printHelp) {
	    	options["help"].set = true;
	    }
    }
	bool isSet(std::string name)    {return ((options.find(name) != options.end()) && options[name].set);}
	std::string getValueAsString(std::string name, std::string defaultValue){
        assert(options.find(name) != options.end());
	    std::string value = options[name].value;
	    return (value != "") ? value : defaultValue;
    }
	int32_t getValueAsInt(std::string name, int32_t defaultValue){
        assert(options.find(name) != options.end());
	    std::string value = options[name].value;
	    if (value != "") {
	    	char* numConvPtr;
	    	int32_t intVal = strtol(value.c_str(), &numConvPtr, 10);
	    	return (intVal > 0) ? intVal : defaultValue;
	    } else {
	    	return defaultValue;
	    }
	    return int32_t();
    }
};
}

namespace globals{
extern structures::commandline_parser commandline_parser;
}