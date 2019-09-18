#include "SettingsManager.h"

char SettingsManager::settingsFile[] = "settings/settings.cfg";

SettingsManager::SettingsManager()
{
	loadSettings(settingsFile);
}

SettingsManager::~SettingsManager()
{
	storeSettings(settingsFile);
}

bool SettingsManager::addSetting(Setting s)
{
	settings[s.id] = s;

	if (settingsType.find(s.type) == settingsType.end()) {
		settingsType[s.type] = std::vector<Setting>();
	}
	settingsType[s.type].push_back(&settings[s.id]);

	return true;
}

bool SettingsManager::deleteSetting(std::string id)
{
	Setting s = settings[id];
	delete[] s.data;
	settings.erase(id);

	int i = 0;
	for (; i < settingsType[s.type].size(); i++) {
		if (settingsType[s.type][i].id == id)
			break;
	}

	settingsType[s.type][i] = settingsType[s.type][settingsType[s.type].size()-1];
	settingsType[s.type].pop_back();

	return true;
}

Setting SettingsManager::getSetting(std::string id)
{
	return setting[id];
}

vector<Setting>* SettingsManager::getSettingsType(std::string type)
{
	return &settingsType[type];
}

void SettingsManager::storeSettings(const char* filename)
{

}

void SettingsManager::loadSettings(const char* filename)
{
	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cout << "Settingsfile was not found." << std::endl;
		return
	}

	while (!file.eof()) {
		//getting the id, type and datasize
		Setting s = {};
		file >> s.id;
		file >> s.type;
		file >> s.byteLength;
		s.data = new char[s.byteLength];
		for (int i = 0; i < s.byteLength; i++) {
			file >> s.data[i];
		}
		addSetting(s);
	}
}
