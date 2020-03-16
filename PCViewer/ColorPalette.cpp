#include "ColorPalette.h"


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

    colorCategories.push_back("div");
    colorCategories.push_back("qual");
    colorCategories.push_back("seq");

    // fill Lists for seq qual etc...
    for(unsigned int i = 0; i < palettes.size(); ++i)
    {
        if (palettes.at(i).category == colorCategories.at(0))
        {
            palettesDiv.push_back((palettes.at(i)));
            divNameList.push_back(palettes.at(i).cName);
        }
        else if (palettes.at(i).category == colorCategories.at(1))
        {
            palettesQual.push_back((palettes.at(i)));
            qualNameList.push_back(palettes.at(i).cName);
        }
        else if (palettes.at(i).category == colorCategories.at(2))
        {
            palettesSeq.push_back((palettes.at(i)));
            seqNameList.push_back(palettes.at(i).cName);
        }

    }
}


ColorPalette::~ColorPalette()
{

}

