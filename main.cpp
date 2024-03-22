#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include "random.h"
#include "PolarCode.h"

struct random rnd;

void get_corrupted_inplace(const std::vector<int> &random_codeword, float variability, std::vector<float> &result) {
    for (int i = 0; i < random_codeword.size(); ++i) {
        result[i] = ((random_codeword[i] * 2) - 1) + sqrt(variability) * rnd.normal(rnd.rng);
    }
}

std::vector<float> get_corrupted(const std::vector<int> &random_codeword, float variability) {
    std::vector<float> ans(random_codeword.size());
    get_corrupted_inplace(random_codeword, variability, ans);
    return ans;
}


bool words_equals(std::vector<int> const &a, std::vector<int> const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

float collectData(PolarCode polarCode, int samples, float variability) {
    int n = 1 << polarCode.m;
    int k = polarCode.k;
    std::vector<int> information_word(n - polarCode.k);
    std::vector<int> coded_word(1 << polarCode.m);
    std::vector<float> corrupted_word(1 << polarCode.m);
    std::vector<int> result_without_freezing;

    volatile float correct = 0;

    float all = samples;
    for (int tries = 0; tries < samples; tries++) {

        rnd.get_random_word_inplace(n - k, information_word);
        polarCode.encode_inplace(information_word, coded_word);
        get_corrupted_inplace(coded_word, variability, corrupted_word);
        auto result = polarCode.decode(corrupted_word);
        polarCode._encode(result, 0, n);
        result_without_freezing.clear();
        for (int i = 0; i < result.size(); ++i) {
            if (!polarCode.polarizedChannel.isFrozen(i)) {
                result_without_freezing.push_back(result[i]);
            }
        }
        if (words_equals(result_without_freezing, information_word)) {
            correct++;
        }
    }
    return (all - correct) / all;
}

void collectDataConsole(int m, int k, int L, int precision) {
    int n = 1 << m;
    std::cout << "Отношение сигнал-шум\t|\tВероятность ошибки\t|\n" <<
                 "------------------------|-----------------------|\n";
    std::cout << std::fixed << std::setprecision(6);
    for (float Eb = 4.5; Eb < 6.0; Eb += 0.5) {
        float variability = static_cast<float>(n) / (2 * pow(10.0, (Eb / 10)) * (n - k));
        float errorProbability = collectData(PolarCode(m, k, L, variability), precision, variability);
        std::cout << "\t\t" << Eb << "\t\t|\t\t" << errorProbability << "\t\t|\n"
        << "------------------------|-----------------------|\n";
    }
}


void collectDataCSV(int m, int k, int L, int precision) {
    int n = 1 << m;
    std::cout << std::fixed << std::setprecision(6);
    for (float Eb = 4.5; Eb < 6.0; Eb += 0.5) {
        float variability = static_cast<float>(n) / (2 * pow(10.0, (Eb / 10)) * (n - k));
        float errorProbability = collectData(PolarCode(m, k, L, variability), precision, variability);
        std::cout << Eb << ";" << errorProbability << "\n";
    }
}

int main() {
    collectDataCSV(6, 28, 4, 100000);
    return 0;
}