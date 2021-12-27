#pragma once

#include "Data.hpp"

/*  This class extends the standard Data class to clustered multiparameter represenatives
*   
*   There are 2 modes this clustered data is available:
*   1)  In mode 1 the cluster amount is given for each data point
*       For each data point has a shared pointer to a single Data which represents the next layer of refinement
*       The base(true data points) is a Data object
*   2)  In mode 2 for each pair of attributes a clustering exists (this includes extra overhead for cluster counts)
*       For each clustering a shared pointer to a single Data which represents the next layer exists
*       Mode 2 has the problem of having the data multiple times in layer 0 -> has highly inflated memory footprint
*
*   Currently focus is on mode1 as it appears to be the better approach concerning complexity and maintanabilty
*/

class ClusterData: public Data
{
public:
    ClusterData(){};
};