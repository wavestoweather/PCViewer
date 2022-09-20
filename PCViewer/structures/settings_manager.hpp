#pragma once
#include <vector>
#include <robin_hood.h>
#include <string>
#include <memory_view.hpp>

namespace structures{
class settings_manager {
public:
	struct setting {
		std::string id;
		std::string type;
		std::vector<uint8_t> storage;

		template<class T>
		T& get(){ assert(sizeof(T) == storage.size()); return *reinterpret_cast<T*>(storage.data());}
		template<class T>
		util::memory_view<T> data(){return util::memory_view<T>(util::memory_view(storage));}

		bool operator==(const setting& other) const{
			return id == other.id;
		}
        operator bool() const{
            return storage.size();
        }
	};

	settings_manager();
	~settings_manager();

	bool add_setting(const setting& s, bool autostore = true);
	bool delete_setting(std::string_view id);
	setting& get_setting(std::string_view id);
	std::vector<setting*>* get_settings_type(std::string type);
	setting notFound{};

private:
	const std::string_view settings_file{"settings.cfg"};
	const std::string_view file_path{""};
	void store_settings(std::string_view filename);
	void load_settings(std::string_view filename);

	robin_hood::unordered_map<std::string, setting> settings;
	robin_hood::unordered_map<std::string, std::vector<setting*>> settings_type;
};

}

namespace globals{
extern structures::settings_manager settings_manager;
}
