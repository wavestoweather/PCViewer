#pragma once
#include <string_view>
#include <memory_view.hpp>

namespace structures{
struct c_file{
    enum class origin{
        set,    // seek from beginning of file
        cur,    // seed from current file pos
        end     // seek from end of file
    };

	c_file(std::string_view filename, std::string_view open_mode): handle(fopen(filename.data(), open_mode.data())){}
	~c_file() {fclose(handle);}
	c_file(const c_file&) = delete;
	c_file& operator=(const c_file&) = delete;
    template<class T>
	size_t read(util::memory_view<T> data) {return fread(data.data(), sizeof(T), data.size());}
    template<class T>
    size_t write(const util::memory_view<T> data) {return fwrite(data.data(), sizeof(T), data.size());}
    int seek(long offset, origin origin) {return fseek(handle, offset, static_cast<int>(origin));}
    long tell() {return ftell(handle);}
	FILE* handle;
};
}