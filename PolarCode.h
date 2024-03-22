#ifndef POLAR_CODES_POLARCODE_H
#define POLAR_CODES_POLARCODE_H

#include <vector>
#include "stack.h"
#include "ChannelAWGN.h"
#include <immintrin.h>

struct PolarCode {
    int m;
    int k;
    int L;
private:
    stack<int> inactivePathIndices;
    std::vector<stack<int>> inactiveArrayIndices;
    std::vector<float> LLRs;
    std::vector<bool> activePath;
    std::vector<std::vector<float *>> arrayPointer_P;
    std::vector<std::vector<int *>> arrayPointer_C;
    std::vector<std::vector<int>> pathIndexToArrayIndex;
    std::vector<std::vector<int>> arrayReferenceCount;
public:
    ChannelAWGN polarizedChannel;
private:
    bool *contForks;
    std::pair<float, int> *find_max_array;
public:
    PolarCode(int m, int k, int list_size, float variability);

    ~PolarCode();

    void _encode(std::vector<int> &result, int l, int r);

    void encode_inplace(const std::vector<int> &information_word, std::vector<int> &result);

    std::vector<int> encode(std::vector<int> &information_word);

    std::vector<int> decode(std::vector<float> &y);

private:
    void initializeDataStructures();

    int assignInitialPath();

    int clonePath(int l);

    void killPath(int l);

    float *getArrayPointer_P(int lambda, int l);

    int *getArrayPointer_C(int lambda, int l);

    bool pathIndexInactive(int l);


    __m256 copy_sign(__m256 srcSign, __m256 srcValue);

    __m256 f_avx(float *pA, float *pB);

    __m256 g_avx(float *pA, float *pB, int *pC);


    void recursivelyCalcP(int lambda, int phi);

    int countPhaseIndex(int phase);

    void recursivelyUpdateC(int phi);

    static int inline sign(float a);

    static float Phi(float mu, float lambda, int u);

    void continuePaths_UnfrozenBit(int phi);
};

#endif //POLAR_CODES_POLARCODE_H
