#ifndef POLAR_CODES_SUPPORT_H
#define POLAR_CODES_SUPPORT_H
//#define DEBUG_MODE
#include <vector>
#include <iostream>

template<typename T>
void printVector(std::vector<T> &a) {
    for (auto &i: a) {
        std::cout << i << ' ';
    }
    std::cout << '\n';
}

#endif //POLAR_CODES_SUPPORT_H
