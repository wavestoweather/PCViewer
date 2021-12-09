#pragma once

#include<vector>

template<class T>
//column major matrix class
class Matrix{
public:
    int nRows, nColumns;
    std::vector<T> data;

    Matrix(int rows, int columns):nRows(rows), nColumns(columns), data(std::vector<T>(nRows * nColumns)){}
    Matrix(int rows, int columns, std::vector<T> d):nRows(rows), nColumns(columns), data(d){}
    
    double& operator()(int row, int column){
        return data[column * nRows + row];
    }
    double& operator()(int row, int column) const{
        return data[column * nRows + row];
    }
private:
};