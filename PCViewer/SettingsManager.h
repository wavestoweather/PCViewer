#ifndef SettingsManager_H
#define SettingsManager_H

#include <vector>
#include <unordered_map>
#include <string>

class SettingsManager {
public:
	struct Setting {
		std::string id;
		std::string type;
		uint32_t byteLength;
		void* data;
	};

	public SettingsManager();
	public ~SettingsManager();

	bool addSetting(Setting s);
	bool deleteSetting(std::string id);
	Setting getSetting(std::string id);
	vector<Setting> getSettingsType(std::string type);

private:
	void storeSettings(const char* filename);
	void loadSettings(const char* filename);

	std::unordered_map<std::string,Setting> settings;
};

#endif // !SettingsManager_H
