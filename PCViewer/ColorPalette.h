#include "colorbrewer/colorbrewer.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include "imgui/imgui.h"

typedef struct {
        std::string cName = std::string("");
        unsigned int maxcolors = 0;
        std::string category = std::string("");
        bool colorblind = false;
        char* categoryC = nullptr;
        std::vector<ImVec4> custColors = std::vector<ImVec4>();
        } CPalette;





class ColorPalette{
public:
    ColorPalette();
    ~ColorPalette();

    std::vector<CPalette> palettes;

    std::vector<CPalette> palettesDiv;
    std::vector<CPalette> palettesQual;
    std::vector<CPalette> palettesSeq;
    std::vector<CPalette> palettesCust;

//    std::vector<std::vector<CPalette>*> palettesMatrix;

    std::vector<std::string> colorCategories;

    char* defaultCategory;

//    std::map<std::string, std::vector<std::string>> colorCatMap;

    std::vector<std::string> divNameList;
    std::vector<std::string> qualNameList;
    std::vector<std::string> seqNameList;
    std::vector<std::string> custNameList;

    std::vector<std::vector<std::string>> paletteNamesVec;

    std::vector<char *> convVecStrToChar(const std::vector<std::string> strVec);

    static char* convStrToChar(const std::string & s);

    CPalette* getPalletteWithName(std::string str);

    CPalette *getPalletteWithNrs(unsigned int cat, unsigned int ipal);

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

    unsigned int skipFirstAttributes;

    int alphaLines;
    int alphaFill;

    bool applyToFillColor;
    bool applyToLineColor;

    bool backupLineColor;
    bool backupFillColor;

    bool bvaluesChanged;

    void backupColors(std::vector<ImVec4> lineColors, std::vector<ImVec4> fillColors);

    // The set methods set valuesChanged to true.
    void setChosenCategoryNr(unsigned int i);
    void setChosenPaletteNr(unsigned int i);
    void setChosenNrColorNr(unsigned int i);
    void setChosenSkipFirstAttributes(unsigned int i);
    void setApplyToFillColor(bool b);
    void setApplyToLineColor(bool b);

    bool getBValuesChanged();


};

