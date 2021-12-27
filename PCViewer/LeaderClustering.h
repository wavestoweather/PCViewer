#pragma once
#include "Data.hpp"

/*
*   Implements r tree accelerated leaders clustering (the leaders are inserted into an r-tree)
*/

class LeaderClustering{
public:
    // applies leaders clustering to data stored in 'in'
    // note that the compression of 'in' by subdimensioning is lost!
    // epsilon is avector of the epsilon parameter scaled with the min max for each dimension
    static void cluster(const Data& in, Data& out, const std::vector<float>& epsilon);
};