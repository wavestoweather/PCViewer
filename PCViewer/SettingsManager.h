#ifndef SettingsManager_H
#define SettingsManager_H

#include <vector>
#include <unordered_map>
#include <string>
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

	public SettingsManager();
	public ~SettingsManager();

	bool addSetting(Setting s);
	bool deleteSetting(std::string id);
	Setting getSetting(std::string id);
	vector<Setting>* getSettingsType(std::string type);

private:
	static char settingsFile[];
	void storeSettings(const char* filename);
	void loadSettings(const char* filename);

	std::unordered_map<std::string,Setting> settings;
	std::unordered_map<std::string, std::vector<Setting*>> settingsType;
};

#endif // !SettingsManager_H
