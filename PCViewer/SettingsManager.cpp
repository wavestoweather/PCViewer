#include "SettingsManager.h"

# include <string>

char SettingsManager::settingsFile[] = "settings.cfg";

SettingsManager::SettingsManager()
{
	loadSettings(settingsFile);
}

SettingsManager::~SettingsManager()
{
	storeSettings(settingsFile);
	std::unordered_map<std::string, Setting> c(settings);
	for (auto& s : c) {
		deleteSetting(s.second.id);
	}
}

bool SettingsManager::addSetting(Setting s, bool autostore)
{
	if (s.id.empty() || s.type.empty()) return false;

	void* data = new char[s.byteLength];
	memcpy(data, s.data, s.byteLength);

	s.data = data;
	bool exists = false;
	if (settings.find(s.id) != settings.end()) {
		delete settings[s.id].data;
		exists = true;
	}
	settings[s.id] = s;

	if (settingsType.find(s.type) == settingsType.end()) {
		settingsType[s.type] = std::vector<Setting*>();
	}

	if(!exists)
		settingsType[s.type].push_back(&settings[s.id]);

	if(autostore)
		storeSettings(settingsFile);

	return true;
}

bool SettingsManager::deleteSetting(std::string id)
{
	Setting s = settings[id];
	delete[] s.data;

	int i = 0;
	for (; i < settingsType[s.type].size(); i++) {
		if (settingsType[s.type][i]->id == id)
			break;
	}

	settingsType[s.type][i] = settingsType[s.type][settingsType[s.type].size()-1];
	settingsType[s.type].pop_back();

	settings.erase(id);
	return true;
}

SettingsManager::Setting SettingsManager::getSetting(std::string id)
{
	if (settings.find(id) == settings.end()) return { "settingnotfound" };
	return settings[id];
}

std::vector<SettingsManager::Setting*>* SettingsManager::getSettingsType(std::string type)
{
	return &settingsType[type];
}

void SettingsManager::storeSettings(const char* filename)
{
	std::ofstream file(filename, std::ifstream::binary);
	for (auto& s : settings) {
		file << "\"" << s.second.id << "\"" << ' ' << "\"" << s.second.type << "\"" << ' ' << s.second.byteLength << ' ';
		for (int i = 0; i < s.second.byteLength; i++) {
			file << ((char*)s.second.data)[i];
		}
		file << "\n";
	}
	file.close();
}

void SettingsManager::loadSettings(const char* filename)
{
	std::ifstream file(filename, std::ifstream::binary);

	if (!file.is_open()) {
		std::cout << "Settingsfile was not found or no settings exist." << std::endl;
		return;
	}

	while (!file.eof()) {
		//getting the id, type and datasize
		Setting s = {};
		file >> s.id;
		if (s.id.size() == 0)
			break;
		if (s.id[0] == '\"') {
			if (!(s.id[s.id.size() - 1] == '\"')) {
				s.id = s.id.substr(1);
				std::string nextWord;
				file >> nextWord;
				while (nextWord[nextWord.size() - 1] != '\"') {
					s.id += " " + nextWord;
					file >> nextWord;
				}
				s.id += " " + nextWord.substr(0, nextWord.size() - 1);
			}
			else {
				s.id = s.id.substr(1, s.id.size() - 2);
			}
		}
		
		file >> s.type;
		if (s.type[0] == '\"') {
			if (!(s.type[s.type.length() - 1] == '\"')) {
				s.type = s.type.substr(1);
				std::string nextWord;
				file >> nextWord;
				while (nextWord[nextWord.size() - 1] != '\"') {
					s.type += " " + nextWord;
					file >> nextWord;
				}
				s.type += " " + nextWord.substr(0, nextWord.size() - 1);
			}
			else {
				s.type = s.type.substr(1, s.type.size() - 2);
			}
		}

		file >> s.byteLength;
		s.data = new char[s.byteLength];
		file.get();
		file.read((char*)s.data,s.byteLength);
		file.get();
		if (s.id.size() != 0)
			addSetting(s, false);

		//for (int i = 0; i < s.byteLength; ++i)
		//	std::cout << (int)((char*)s.data)[i];
		
		delete[] s.data;
	}

	file.close();
}
