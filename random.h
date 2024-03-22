#ifndef POLAR_CODES_RANDOM_H
#define POLAR_CODES_RANDOM_H

#include <random>
#include <chrono>

struct random {
    std::mt19937 rng;
    std::uniform_int_distribution<int> uniform_int;
    std::uniform_real_distribution<> uniform_real;
    std::normal_distribution<float> normal;

    random();
    int rnd(int l, int r);
    std::vector<int> get_random_word(int length);
    void get_random_word_inplace(int length, std::vector<int> &a);
};

#endif //POLAR_CODES_RANDOM_H
