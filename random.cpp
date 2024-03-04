//
// Created by dis3e on 03.11.23.
//
#include "random.h"

random::random() : rng((std::chrono::steady_clock::now().time_since_epoch().count())),
                   uniform_int(0, std::numeric_limits<int>::max()),
                   uniform_real(0.0, 1.0),
                   normal(0, 1) {

}

int random::rnd(int l, int r) {
    return (uniform_int(rng) % (r - l + 1) + l);
}

std::vector<int> random::get_random_word(int length) {
    std::vector<int> result(length);
    get_random_word_inplace(length, result);
    return result;
}


void random::get_random_word_inplace(int length, std::vector<int>& a) {
    for (auto &t: a) t = rnd(0, 1);
}