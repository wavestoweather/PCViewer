#include "colorbrewer/colorbrewer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "imgui/imgui.h"

struct CPalette{
        std::string cName;
        int maxcolors;
        std::string category;
        bool colorblind;
        char* categoryC;
        };





class ColorPalette{
public:
    ColorPalette();
    ~ColorPalette();

    std::vector<CPalette> palettes;

    std::vector<CPalette> palettesDiv;
    std::vector<CPalette> palettesQual;
    std::vector<CPalette> palettesSeq;

    std::vector<std::string> colorCategories;

    char* defaultCategory;

//    std::map<std::string, std::vector<std::string>> colorCatMap;

    std::vector<std::string> divNameList;
    std::vector<std::string> qualNameList;
    std::vector<std::string> seqNameList;

    std::vector<std::vector<std::string>> paletteNamesVec;

    std::vector<char *> convVecStrToChar(const std::vector<std::string> strVec);

    static char* convStrToChar(const std::string & s);

    CPalette* getPalletteWithName(std::string str);

    std::vector<ImVec4> getPallettAsImVec4(unsigned int categoryNr, unsigned int paletteNr, unsigned int nrColors, float alpha = 0.4);



protected:

private:

};


class ColorPaletteManager{

public:
    ColorPaletteManager();
    ~ColorPaletteManager();

    ColorPalette colorPalette;

    bool useColorPalette;

    unsigned int chosenCategoryNr;
    unsigned int chosenPaletteNr;
    unsigned int chosenNrColorNr;


};

