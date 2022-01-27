#include "HirarchyCreation.hpp"

#include "HirarchyNode.hpp"
#include "../rTree/RTreeDynamic.h"
#include <filesystem>
#include <iostream>
#include <fstream>
#include <map>
#include <queue>

namespace compression
{
    void createHirarchy(const std::string_view& outputFolder, DataLoader& loader, float lvl0eps, int levels, int lvlMultiplier) 
    {
        createTempHirarchy(outputFolder, loader, lvl0eps, levels, lvlMultiplier);
        compressTempHirarchy(outputFolder);
    }

    void createTempHirarchy(const std::string_view& outputFolder, DataLoader& loader, float lvl0eps, int levels, int lvlMultiplier) 
    {
        try{
            std::filesystem::current_path(outputFolder);
            //creating the temp folder for the temporary non compressed files
            std::filesystem::create_directory("temp/");

            std::vector<float> dataPoint;
            bool hasData = loader.getNextNormalized(dataPoint);
            if(!hasData){
                std::cout << "compression::createHirarchy(...) given loader has either no elements or was already loaded. Reset or insert loader with data elements" << std::endl;
                return;
            }

            // converting lvl multiplier to epsilon multiplier
            double epsMult = pow(1.0/lvlMultiplier, 1.0/dataPoint.size());
            HirarchyNode root(dataPoint, lvl0eps, epsMult, 0, levels);   //constructor automatically inserts the first data point

            while(loader.getNextNormalized(dataPoint)){
                //insert into the hirarchy
                root.addDataPoint(dataPoint);

                //should add caching strategies to avoid memory overflow and inbetween writeouts
            }

            //final writeout to disk
        }
        catch(std::filesystem::filesystem_error err){
            std::cout << "Error trying to open output folder " << err.path1() << " with code: " << err.code() << std::endl;
        }
    }
    
    void compressTempHirarchy(const std::string_view& outputFolder) 
    {
        
    }
}