#pragma once

#include <memory>
#include <atomic>
#include "ClusterData.hpp"

/*
*   Class which opens a large scale dataset in an extra thread, clusters it and provides loaded data
*   in a ClusterData shared pointer when ready
*
*   Objects of this class act in a 2 stage process:
*   1)  Construction/Configuration stage starts upon creating an instance
*   2)  Async analysation and loading of large scale data is started upon calling dispatch()
*/

class DataClusteringLoader{
public:

    std::atomic<float> progress = 0;            //atomic float value to easily update the progress in a thread safe manner
    std::shared_ptr<ClusterData> clusterData;
private:
};