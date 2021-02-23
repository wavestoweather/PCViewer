#ifndef SettingsManager_H
#define SettingsManager_H

#include <vector>
#include <unordered_map>
#include <string.h>
#include <fstream>
#include <iostream>

class SettingsManager {
public:
	struct Setting {
		std::string id;
		std::string type;
		uint32_t byteLength;
		void* data;
	};

	SettingsManager();
	~SettingsManager();

	bool addSetting(Setting s, bool autostore = true);
	bool deleteSetting(std::string id);
	Setting getSetting(std::string id);
	std::vector<Setting*>* getSettingsType(std::string type);

private:
	static char settingsFile[];
	static char filePath[];
	void storeSettings(const char* filename);
	void loadSettings(const char* filename);

	std::unordered_map<std::string,Setting> settings;
	std::unordered_map<std::string, std::vector<Setting*>> settingsType;
};

#endif // !SettingsManager_H
