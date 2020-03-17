#include "ColorPalette.h"

#include <string.h>
#include <algorithm>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

ColorPalette::ColorPalette()
{



    // Diverging colormaps
    palettes.push_back(CPalette{std::string("BrBG"), 11, std::string("div"), true});
    palettes.push_back(CPalette{std::string("PiYG"), 11, std::string("div"), true});
    palettes.push_back(CPalette{std::string("PRGn"), 11, std::string("div"), true});
    palettes.push_back(CPalette{std::string("PuOr"), 11, std::string("div"), true});
    palettes.push_back(CPalette{std::string("RdBu"), 11, std::string("div"), true});
    palettes.push_back(CPalette{std::string("RdGy"), 11, std::string("div"), false});
    palettes.push_back(CPalette{std::string("RdYlBu"), 11, std::string("div"), true});
    palettes.push_back(CPalette{std::string("RdYlGn"), 11, std::string("div"), false});
    palettes.push_back(CPalette{std::string("Spectral"), 11, std::string("div"), false});

    // qualitative colormaps
    palettes.push_back(CPalette{std::string("Accent"), 8, std::string("qual"), false});
    palettes.push_back(CPalette{std::string("Dark2"), 8, std::string("qual"), true});
    palettes.push_back(CPalette{std::string("Paired"), 12, std::string("qual"), true});
    palettes.push_back(CPalette{std::string("Pastel1"), 9, std::string("qual"), false});
    palettes.push_back(CPalette{std::string("Pastel2"), 8, std::string("qual"), false});
    palettes.push_back(CPalette{std::string("Set1"), 9, std::string("qual"), false});
    palettes.push_back(CPalette{std::string("Set2"), 8, std::string("qual"), true});
    palettes.push_back(CPalette{std::string("Set3"), 12, std::string("qual"), false});

    // sequential colormaps
    palettes.push_back(CPalette{std::string("Blues"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("BuGn"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("BuPu"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("GnBu"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("Greens"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("Greys"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("Oranges"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("OrRd"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("PuBu"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("PuBuGn"), 9, std::string("seq"), true});

    palettes.push_back(CPalette{std::string("PuRd"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("Purples"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("RdPu"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("Reds"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("YlGn"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("YlGnBu"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("YlOrBr"), 9, std::string("seq"), true});
    palettes.push_back(CPalette{std::string("YlOrRd"), 9, std::string("seq"), true});

//    palettes.push_back(CPalette{std::string("Accent"), 9, std::string("seq"), true});
    std::vector<ImVec4> defaultColorVec;
    defaultColorVec.push_back(ImVec4(.4,.4,.4,.4));
    palettes.push_back(CPalette{std::string("Default"), 1, std::string("cust"), false, nullptr, defaultColorVec});

    colorCategories.push_back("div");
    colorCategories.push_back("qual");
    colorCategories.push_back("seq");
    colorCategories.push_back("cust");

//    colorCategoriesC = {"div","qual","seq"};

    // fill Lists for seq qual etc...
    for(unsigned int i = 0; i < palettes.size(); ++i)
    {
        palettes[i].categoryC = &palettes[i].category[0];
        if (palettes.at(i).category == colorCategories.at(0))
        {
            palettesDiv.push_back((palettes[i]));
            divNameList.push_back(palettes.at(i).cName);
        }
        else if (palettes.at(i).category == colorCategories.at(1))
        {
            palettesQual.push_back((palettes[i]));
            qualNameList.push_back(palettes.at(i).cName);
        }
        else if (palettes.at(i).category == colorCategories.at(2))
        {
            palettesSeq.push_back((palettes[i]));
            seqNameList.push_back(palettes.at(i).cName);
        }
        else if (palettes.at(i).category == colorCategories.at(3))
        {
            palettesCust.push_back((palettes[i]));
            custNameList.push_back(palettes.at(i).cName);
        }

    }


    paletteNamesVec.push_back(divNameList);
    paletteNamesVec.push_back(qualNameList);
    paletteNamesVec.push_back(seqNameList);
    paletteNamesVec.push_back(custNameList);

    // Does not work once the object is copied...
//    palettesMatrix.push_back(&palettesDiv);
//    palettesMatrix.push_back(&palettesQual);
//    palettesMatrix.push_back(&palettesSeq);
//    palettesMatrix.push_back(&palettesCust);

}


ColorPalette::~ColorPalette()
{

}

// ToDo: This causes memory leaks...
std::vector<char*> ColorPalette::convVecStrToChar(const std::vector<std::string> strVec)
{
    std::vector<char*>  vc;
    std::transform(strVec.begin(), strVec.end(), std::back_inserter(vc), ColorPalette::convStrToChar);
    return vc;

}

char* ColorPalette::convStrToChar(const std::string & s)
{
    std::unique_ptr<char[]> pc(new char[s.size()+1]); // = new char[s.size()+1];
    strcpy(pc.get(), s.c_str());
    return pc.get();
}


CPalette* ColorPalette::getPalletteWithName(std::string str)
{
    for (unsigned int i = 0; i < this->palettes.size(); ++i)
    {
        if (this->palettes.at(i).cName == str)
        {
            return &(this->palettes[i]);
        }
    }
}


CPalette *ColorPalette::getPalletteWithNrs(unsigned int cat, unsigned int ipal)
{
    std::vector<CPalette> *ptrCPallList = nullptr;
    switch (cat){
        case 0:
            ptrCPallList = &palettesDiv;
            break;
        case 1:
            ptrCPallList = &palettesQual;
            break;
        case 2:
            ptrCPallList = &palettesSeq;
            break;
        case 3:
            ptrCPallList = &palettesCust;
            break;
    }
    return &ptrCPallList->at(ipal);
}


std::vector<ImVec4> ColorPalette::getPallettAsImVec4(unsigned int categoryNr ,unsigned int paletteNr, unsigned int nrColors, float alpha)
{
    std::vector<ImVec4> choosenColorsImVec;

    const std::string paletteStr = paletteNamesVec.at(categoryNr).at(paletteNr);
    if (categoryNr == 3) // Custom colors
    {
        // retrieve the palette
        CPalette *currPalette = &palettesCust[paletteNr];
        for (unsigned int i = 0; i < nrColors; ++i){
            choosenColorsImVec.push_back(currPalette->custColors[i]);
        }
    }
    else
    {
        unsigned int minVal = 3;

        int numberOfColors = std::min(std::max(minVal,nrColors), getPalletteWithNrs(categoryNr, paletteNr)->maxcolors);

        std::vector<std::string> choosenColors = (brew<std::string>(paletteStr, numberOfColors));

        for (unsigned int i = 0; i < std::min(nrColors,(unsigned int)numberOfColors); ++i){
            int r, g, b;
            r = std::strtol(choosenColors[i].substr(1,2).c_str(), NULL,16);
            g = std::strtol(choosenColors[i].substr(3,2).c_str(), NULL,16);
            b = std::strtol(choosenColors[i].substr(5,2).c_str(), NULL,16);

            choosenColorsImVec.push_back(ImVec4(r/255.,g/255.,b/255.,alpha));
        }
    }

    return choosenColorsImVec;
}

// ##############################

ColorPaletteManager::ColorPaletteManager():
    useColorPalette(true),
    chosenCategoryNr(0),
    chosenPaletteNr(0),
    chosenNrColorNr(0),
    alphaLines(255),
    alphaFill(127),
    applyToFillColor(true),
    applyToLineColor(true),
    backupLineColor(false),
    backupFillColor(false),
    bvaluesChanged(false)
{

}

ColorPaletteManager::~ColorPaletteManager()
{

}


void ColorPaletteManager::setChosenCategoryNr(unsigned int i)
{
    chosenCategoryNr = i;
    bvaluesChanged = true;

}


void ColorPaletteManager::setChosenPaletteNr(unsigned int i)
{
    chosenPaletteNr = i;
    bvaluesChanged = true;
}


void ColorPaletteManager::setChosenNrColorNr(unsigned int i)
{
    chosenNrColorNr = i;
    bvaluesChanged = true;
}


void ColorPaletteManager::setApplyToFillColor(bool b)
{
    applyToFillColor = b;
    bvaluesChanged = true;
}


void ColorPaletteManager::setApplyToLineColor(bool b)
{
    applyToLineColor = b;
    bvaluesChanged = true;
}





bool ColorPaletteManager::getBValuesChanged()
{
    bool b = bvaluesChanged;
    bvaluesChanged = false;
    return b;

}


void ColorPaletteManager::backupColors(std::vector<ImVec4> lineColors, std::vector<ImVec4> fillColors)
{
    if (backupLineColor)
    {
        backupLineColor = false;
        std::string currColorPaletteName = "line_" + std::to_string(colorPalette.palettesCust.size());
        CPalette currPalette;
        currPalette.cName = currColorPaletteName;
        currPalette.category = "cust";
        currPalette.maxcolors = lineColors.size();
        currPalette.categoryC = &currPalette.category[0];
        currPalette.colorblind = false;
        currPalette.custColors = lineColors;

        this->colorPalette.palettes.push_back(currPalette);
        this->colorPalette.palettesCust.push_back(currPalette);
        this->colorPalette.custNameList.push_back(currPalette.cName);

//        std::vector<std::string> existingNames = this->colorPalette.custNameList;
//        existingNames.push_back(currColorPaletteName);
        this->colorPalette.paletteNamesVec[3].push_back(currColorPaletteName);

    }
    if (backupFillColor)
    {
        backupFillColor = false;
        std::string currColorPaletteName = "fill_" + std::to_string(colorPalette.palettesCust.size());
        CPalette currPalette;
        currPalette.cName = currColorPaletteName;
        currPalette.category = "cust";
        currPalette.maxcolors = fillColors.size();
        currPalette.categoryC = &currPalette.category[0];
        currPalette.colorblind = false;
        currPalette.custColors = fillColors;

        this->colorPalette.palettes.push_back(currPalette);
        this->colorPalette.palettesCust.push_back(currPalette);
        this->colorPalette.custNameList.push_back(currPalette.cName);

        this->colorPalette.paletteNamesVec[3].push_back(currColorPaletteName);
    }

}


