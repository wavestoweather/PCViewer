#include "colorbrewer/colorbrewer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>

struct CPalette{
        std::string cName;
        int maxcolors;
        std::string category;
        };

class ColorPalette{
public:
    ColorPalette();
    ~ColorPalette();

    std::vector<CPalette> palettes;

    std::vector<std::string> colorCategories;

    std::map<std::string, std::vector<std::string>> colorCatMap;


protected:

private:

};

// QList<QColor> qColors(brew<QColor>("BuGn", 3));
