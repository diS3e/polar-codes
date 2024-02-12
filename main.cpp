//
// Created by dis3e on 07.12.23.
//
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <iomanip>
#include "random.h"
#include<stack>
#include <immintrin.h>
#include "support.h"
//Привести код в приличный вид
//

struct random rnd;

//Freezed subchannels
std::set<int> F;
std::vector<std::pair<float, int>> error_probability;

static float theta(float x) {
    if (x > 12) {
        return 0.98611 * x - 2.31515;
    } else if (3.5 < x && x <= 12) {
        return x * (0.0090047 * x + 0.76943) - 0.95068;
    } else if (1 < x && x <= 3.5) {
        return x * (0.062883 * x + 0.36784) - 0.16267;
    } else {
        return x * (0.22024 * x + 0.06448);
    }
}

static float Q(float x) {
    return erfc(x / sqrt(2)) / 2;
}

float finding_E(int lambda, int i, float variability) {
    if (lambda == 0 && i == 0) {
        return 2 / variability;
    }

    if ((i & 1) == 0) {
        return theta(finding_E(lambda - 1, i >> 1, variability));
    } else {
        return 2 * finding_E(lambda - 1, i >> 1, variability);
    }
}

float finding_error_probability_awgn(int _m, int i, float variability) {
    return Q(sqrt(finding_E(_m, i, variability) / 2.0));
}

void find_freezed_channels(int m, int k, float variability) {
    for (int i = 0; i < (1 << m); ++i) {
        error_probability.emplace_back(finding_error_probability_awgn(m, i, variability), i);
    }

    auto comparator = [](const auto &a, const auto &b) { return a.first > b.first; };
    std::sort(error_probability.begin(), error_probability.end(), comparator);

    for (int i = 0; i < k; ++i) {
        F.insert(error_probability[i].second);
    }
}

struct SCL {
    int m;
    int k;
    int L;
    std::vector<int> inactivePathIndices;
    std::vector<float> LLRs;
    std::vector<bool> activePath;
    std::vector<std::vector<float *>> arrayPointer_P;
    std::vector<std::vector<int *>> arrayPointer_C;
    std::vector<std::vector<int>> pathIndexToArrayIndex;
    std::vector<std::vector<int>> inactiveArrayIndices;
    std::vector<std::vector<int>> arrayReferenceCount;


//    std::vector<int> reversed_permutation;

    SCL(int m, int k, int list_size) : m(m), k(k), L(list_size) {

    }

    void _encode(std::vector<int> &result, int l, int r) {
        if (r - l == 1) {
            return;
        }

        for (int i = l; i < l + ((r - l) >> 1); i++) {
            result[i] ^= result[((r - l) >> 1) + i];
        }
        _encode(result, l, l + ((r - l) >> 1));
        _encode(result, l + ((r - l) >> 1), r);
    }

    std::vector<int> encode(std::vector<int> &information_word) {
        std::vector<int> result(1 << m, 0);
        for (int i = 0, j = 0; i < (1 << m); i++) {
            if (F.count(i) == 0) {
                result[i] = information_word[j];
                j++;
            }
        }
        _encode(result, 0, 1 << m);
        return result;
    }

    void initializeDataStructures() {
        inactivePathIndices.resize(L);
        inactiveArrayIndices.resize(m + 1);
        for (int i = 0; i < m + 1; ++i) {
            inactiveArrayIndices[i].resize(L);
        }
        LLRs.resize(L, 0);
        activePath.resize(L, false);
        arrayPointer_P.resize(m + 1, std::vector<float *>(L));
        arrayPointer_C.resize(m + 1, std::vector<int *>(L));
        pathIndexToArrayIndex.resize(m + 1, std::vector<int>(L, 0));
        arrayReferenceCount.resize(m + 1, std::vector<int>(L, 0));

        for (int lambda = 0; lambda < m + 1; ++lambda) {
            for (int s = 0; s < L; ++s) {
                arrayPointer_P[lambda][s] = new float[1 << (m - lambda)]();
                arrayPointer_C[lambda][s] = new int[2 * (1 << (m - lambda))]();
                for (int i = 0; i < 2 * (1 << (m - lambda)); ++i) {
                    arrayPointer_C[lambda][s][i] = -239;
                }

                inactiveArrayIndices[lambda][s] = s;
                pathIndexToArrayIndex[lambda][s] = 0;
                arrayReferenceCount[lambda][s] = 0;
            }
        }
        for (int l = 0; l < L; ++l) {
            activePath[l] = false;
            inactivePathIndices[l] = l;
            LLRs[l] = 0;
        }
    }

    static int poptop(std::vector<int> &a) {
        int v = a[a.size() - 1];
        a.pop_back();
        return v;
    }

    int assignInitialPath() {
        int l = poptop(inactivePathIndices);
        activePath[l] = true;
        for (int lambda = 0; lambda < m + 1; ++lambda) {
            int s = poptop(inactiveArrayIndices[lambda]);
            pathIndexToArrayIndex[lambda][l] = s;
            arrayReferenceCount[lambda][s] = 1;
        }
        return l;
    }

    int clonePath(int l) {
        int l1 = poptop(inactivePathIndices);
        activePath[l1] = true;
        LLRs[l1] = LLRs[l];
        for (int lambda = 0; lambda < m + 1; ++lambda) {
            int s = pathIndexToArrayIndex[lambda][l];
            pathIndexToArrayIndex[lambda][l1] = s;
            arrayReferenceCount[lambda][s]++;
        }
        return l1;
    }

    void killPath(int l) {
        activePath[l] = false;
        inactivePathIndices.push_back(l);
        LLRs[l] = 0;
        for (int lambda = 0; lambda < m + 1; ++lambda) {
            int s = pathIndexToArrayIndex[lambda][l];
            arrayReferenceCount[lambda][s]--;
            if (arrayReferenceCount[lambda][s] == 0) {
                inactiveArrayIndices[lambda].push_back(s);
            }
        }
    }

    float *getArrayPointer_P(int lambda, int l) {
        int s = pathIndexToArrayIndex[lambda][l];
        int s1;
        if (arrayReferenceCount[lambda][s] == 1) {
            s1 = s;
        } else {
            s1 = poptop(inactiveArrayIndices[lambda]);
            std::copy(arrayPointer_C[lambda][s], arrayPointer_C[lambda][s] + (1 << (m - lambda + 1)),
                      arrayPointer_C[lambda][s1]);
            std::copy(arrayPointer_P[lambda][s], arrayPointer_P[lambda][s] + (1 << (m - lambda)),
                      arrayPointer_P[lambda][s1]);
            arrayReferenceCount[lambda][s]--;
            arrayReferenceCount[lambda][s1] = 1;
            pathIndexToArrayIndex[lambda][l] = s1;

        }
        return arrayPointer_P[lambda][s1];
    }

    int *getArrayPointer_C(int lambda, int l) {
        int s = pathIndexToArrayIndex[lambda][l];
        int s1;
        if (arrayReferenceCount[lambda][s] == 1) {
            s1 = s;
        } else {

            s1 = poptop(inactiveArrayIndices[lambda]);

            std::copy(arrayPointer_C[lambda][s], arrayPointer_C[lambda][s] + (1 << (m - lambda + 1)),
                      arrayPointer_C[lambda][s1]);
            std::copy(arrayPointer_P[lambda][s], arrayPointer_P[lambda][s] + (1 << (m - lambda)),
                      arrayPointer_P[lambda][s1]);

            arrayReferenceCount[lambda][s]--;
            arrayReferenceCount[lambda][s1] = 1;
            pathIndexToArrayIndex[lambda][l] = s1;
        }
        return arrayPointer_C[lambda][s1];
    }

    bool pathIndexInactive(int l) {
        if (activePath[l]) {
            return false;
        } else {
            return true;
        }
    }


    __m256 copy_sign(__m256 srcSign, __m256 srcValue) {
        // Extract the signed bit from srcSign
        const __m256 mask0 = _mm256_set1_ps(-0.);
        __m256 tmp0 = _mm256_and_ps(srcSign, mask0);

        // Extract the number without sign of srcValue (abs(srcValue))
        __m256 tmp1 = _mm256_andnot_ps(mask0, srcValue);

        // Merge signed bit with number and return
        return _mm256_or_ps(tmp0, tmp1);
    }

    __m256 f_avx(float *pA, float *pB) {
        __m256 vA = _mm256_loadu_ps(pA);
        __m256 vB = _mm256_loadu_ps(pB);

        const __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi8(0x40));
        __m256 abs_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        __m256 mA = _mm256_and_ps(vA, abs_mask);
        __m256 mB = _mm256_and_ps(vB, abs_mask);
        __m256 mR = _mm256_min_ps(mA, mB);
        __m256 sS = _mm256_xor_ps(vA, vB);

        __m256 sR = _mm256_or_ps(sS, mask);
        __m256 result = copy_sign(sR, mR);
        return result;

    }

    __m256 g_avx(float *pA, float *pB, int *pC) {
        __m256 vA = _mm256_loadu_ps(pA);
        __m256 vB = _mm256_loadu_ps(pB);
        __m256 vC = _mm256_loadu_ps((float *) pC);
        __m256 signs = copy_sign(vC, _mm256_set1_ps(1.f));


        return _mm256_sub_ps(vB, _mm256_mul_ps(signs, vA));

    }


    void recursivelyCalcP(int lambda, int phi) {
        if (lambda == 0) {
            return;
        }
        int psi = phi >> 1;
        if (phi % 2 == 0) {
            recursivelyCalcP(lambda - 1, psi);
        }
        for (int l = 0; l < L; ++l) {
            if (pathIndexInactive(l)) {
                continue;
            }
            auto P_lambda = getArrayPointer_P(lambda, l);
            auto P_lambda1 = getArrayPointer_P(lambda - 1, l);
            auto C_lambda = getArrayPointer_C(lambda, l);
            int layer_size = 1 << (m - lambda);
            for (int beta = 0; beta < (1 << (m - lambda)); beta += 8) {
                if (phi % 2 == 0) {
                    if (layer_size < 8) {
                        for (int i = 0; i < 8 && i < layer_size; ++i) {
                            int beta1 = beta + i;
                            int left = beta1;
                            int right = beta1 + (1 << (m - lambda));
                            float a = P_lambda1[left];
                            float b = P_lambda1[right];
                            P_lambda[beta1] =
                                    std::min(std::abs(a), std::abs(b)) *
                                    (a > 0 ? 1.0 : -1.0) * (b > 0 ? 1.0 : -1.0);
                        }
                    } else {
                        auto res = f_avx(&P_lambda1[beta], &P_lambda1[beta + layer_size]);
                        _mm256_storeu_ps(&(P_lambda[beta]), res);
                    }
                } else {
                    if (layer_size < 8) {
                        for (int i = 0; i < 8 && i < layer_size; ++i) {
                            int beta1 = beta + i;
                            int left = beta1;
                            int right = beta1 + (1 << (m - lambda));
                            float a = P_lambda1[left];
                            float b = P_lambda1[right];
                            int res = beta1;
                            int index = res;
                            int u1 = C_lambda[index];
                            P_lambda[beta1] = b - u1 * a;
                        }
                    } else {
                        auto res = g_avx(&P_lambda1[beta], &P_lambda1[beta + layer_size], &C_lambda[beta]);
                        _mm256_storeu_ps(&(P_lambda[beta]), res);
                    }
                }
            }
        }
    }

//TODO удалить копирование в updateC
    void recursivelyUpdateC(int lambda, int phi) {
//        TODO переписать без рекурсии(можно ксорить напрямую)
        int psi = phi >> 1;
        for (int l = 0; l < L; ++l) {
            if (pathIndexInactive(l)) {
                continue;
            }
            auto C_lambda = getArrayPointer_C(lambda, l);
            auto C_lambda1 = getArrayPointer_C(lambda - 1, l);
            for (int beta = 0; beta < (1 << (m - lambda)); ++beta) {
                if (psi % 2 == 0) {
                    C_lambda1[beta] = - (C_lambda[beta] * C_lambda[beta + (1 << (m - lambda))]);
                    C_lambda1[beta + (1 << (m - lambda))] = C_lambda[beta + (1 << (m - lambda))];
                } else {
                    C_lambda1[beta + (1 << (m - lambda + 1))] = - (C_lambda[beta] * C_lambda[beta + (1 << (m - lambda))]);
                    C_lambda1[beta + (1 << (m - lambda)) + (1 << (m - lambda + 1))] = C_lambda[beta +
                                                                                               (1 << (m - lambda))];
                }
            }
        }
        if (psi % 2 == 1) {
            recursivelyUpdateC(lambda - 1, psi);
        }
    }

    static int inline sign(float a) {
        if (a > 0) {
            return 1;
        } else {
            return -1;
        }
    }

    static float Phi(float mu, float lambda, int u) {
        if (2 * u == (1 - sign(lambda))) {
            return mu;
        } else {
            return mu + std::abs(lambda);
        }
    }


    void continuePaths_UnfrozenBit(int phi) {
        float probForks[L][2];
        int i = 0;
        for (int l = 0; l < L; ++l) {
            if (activePath[l]) {
                auto P_m = getArrayPointer_P(m, l);
                probForks[l][0] = Phi(LLRs[l], P_m[0], 0);
                probForks[l][1] = Phi(LLRs[l], P_m[0], 1);
                i++;
            } else {
                probForks[l][0] = std::numeric_limits<float>::max();
                probForks[l][1] = std::numeric_limits<float>::max();
            }
        }
        int rho = std::min(2 * i, L);


        bool contForks[L][2];
        for (int l = 0; l < L; ++l) {
            contForks[l][0] = false;
            contForks[l][1] = false;
        }

        std::vector<std::pair<float, std::pair<int, int>>> find_max_array(L * 2);
        for (int j = 0; j < L; j++) {
            find_max_array[2 * j] = {probForks[j][0], {j, 0}};
            find_max_array[2 * j + 1] = {probForks[j][1], {j, 1}};
        }
        std::nth_element(find_max_array.begin(), find_max_array.begin() + rho - 1, find_max_array.end());
        for (int j = 0; j < rho; ++j) {
            auto reliable_path = find_max_array[j].second;
            contForks[reliable_path.first][reliable_path.second] = true;
        }
        for (int l = 0; l < L; ++l) {
            if (pathIndexInactive(l)) {
                continue;
            }
            if (!contForks[l][0] && !contForks[l][1]) {
                killPath(l);
            }
        }
        for (int l = 0; l < L; ++l) {
            if (!contForks[l][0] && !contForks[l][1]) {
                continue;
            }
            auto C_m = getArrayPointer_C(m, l);
            if (contForks[l][0] && contForks[l][1]) {
                C_m[(phi % 2)] = -1;

                int l1 = clonePath(l);
                C_m = getArrayPointer_C(m, l1);
                C_m[phi % 2] = 1;
                auto P_m = getArrayPointer_P(m, l);
                LLRs[l] = Phi(LLRs[l], P_m[0], 0);
                P_m = getArrayPointer_P(m, l1);
                LLRs[l1] = Phi(LLRs[l1], P_m[0], 1);
            } else {
                if (contForks[l][0]) {
                    C_m[(phi % 2)] = -1;
                    auto P_m = getArrayPointer_P(m, l);
                    LLRs[l] = Phi(LLRs[l], P_m[0], 0);

                } else {
                    C_m[(phi % 2)] = 1;
                    auto P_m = getArrayPointer_P(m, l);
                    LLRs[l] = Phi(LLRs[l], P_m[0], 1);
                }
            }
        }
    }

    std::vector<int> decode(std::vector<float> &y) {
        initializeDataStructures();
        int l = assignInitialPath();
        auto P_0 = getArrayPointer_P(0, l);
        for (int beta = 0; beta < (1 << m); ++beta) {
            P_0[beta] = -y[beta];
        }
        for (int phi = 0; phi < (1 << m); ++phi) {
            recursivelyCalcP(m, phi);
            if (F.count(phi) != 0) {
                for (int i = 0; i < L; ++i) {
                    if (!activePath[i]) {
                        continue;
                    }
                    auto C_m = getArrayPointer_C(m, i);
                    C_m[phi % 2] = -1;
                    auto P_m = getArrayPointer_P(m, i);
                    LLRs[i] = Phi(LLRs[i], P_m[0], 0);

                }
            } else {
                continuePaths_UnfrozenBit(phi);
            }
            if (phi % 2 == 1) {
                recursivelyUpdateC(m, phi);
            }
#ifdef DEBUG_MODE
            std::cout << "P: ";


            for (int lambda = 0; lambda < m + 1; ++lambda) {

                for (int s = 0; s < L; ++s) {
                    for (int j = 0; j < (1 << (m - lambda)); ++j) {

                        std :: cout << arrayPointer_P[lambda][s][j] << ' ';
                    }
                    std:: cout << "| ";
                }
            }
            std::cout << '\n';
            std::cout << "C: ";


            for (int lambda = 0; lambda < m + 1; ++lambda) {

                for (int s = 0; s < L; ++s) {
                    for (int j = 0; j <  (1 << (m - lambda + 1)); ++j) {
                        if (arrayPointer_C[lambda][s][j] == -1) {
                            std::cout << "_ ";
                        } else
                        std :: cout << arrayPointer_C[lambda][s][j] << ' ';
                    }
                    std:: cout << "| ";
                }
            }
            std::cout << "\n------------------------------------\n";
#endif
        }

        int l1 = 0;
        float p1 = std::numeric_limits<float>::max();
        for (int i = 0; i < L; ++i) {
            if (!activePath[i]) {
                continue;
            }
            if (p1 > LLRs[i]) {
                p1 = LLRs[i];
                l1 = i;
            }
        }
        auto C_0 = getArrayPointer_C(0, l1);
        std::vector<int> res(1 << m);
        for (int i = 0; i < (1 << m); ++i) {
            res[i] = (C_0[i] == 1) ? 1 : 0;
        }
        for (int lambda = 0; lambda < m + 1; ++lambda) {
            for (int i = 0; i < L; ++i) {
                delete[] arrayPointer_C[lambda][i];
                delete[] arrayPointer_P[lambda][i];
            }
        }
        return res;
    }

    std::vector<float> get_corrupt(std::vector<int> &random_codeword, float arg) const {
        int n = 1 << m;
        std::vector<float> ans(n, 0);
        for (int i = 0; i < n; ++i) {
            ans[i] = ((random_codeword[i] * 2) - 1) +
                     sqrt(static_cast<float>(n) / (2 * arg * (n - k))) * rnd.normal(rnd.rng);
        }
        return ans;
    }
};

bool words_equals(std::vector<int> const &a, std::vector<int> const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) return false;
    }
    return true;
}

void collect_data_scl(int m, int k, int L, int precision) {
    for (float Eb = -1.0; Eb < 6.0; Eb += 0.5) {

        float variability = static_cast<float>((1 << m)) / (2 * pow(10.0, (Eb / 10)) * ((1 << m) - k));
        find_freezed_channels(m, k, variability);
        volatile float correct = 0;

        float all = precision;
//#pragma omp parallel for reduction(+:correct, all)
        for (int tries = 0; tries < precision; tries++) {
            SCL scl(m, k, L);

            auto information_word = rnd.get_random_word((1 << scl.m) - scl.k);
            auto coded_word = scl.encode(information_word);
            std::vector<float> corrupt = scl.get_corrupt(coded_word, pow(10.0, (Eb / 10)));
            auto result = scl.decode(corrupt);
            scl._encode(result, 0, 1 << scl.m);

            std::vector<int> result_without_freezing;
            for (int i = 0; i < result.size(); ++i) {
                if (F.count(i) == 0) {
                    result_without_freezing.push_back(result[i]);
                }
            }

            if (words_equals(result_without_freezing, information_word)) {
                correct++;
            } else {
                int x = 5;
                x += 1;
            }
        }
        std::cout << std::fixed << std::setprecision(6) << Eb << ";" << (all - correct) / all << '\n';
    }
}

template<typename T>
void printVector(std::vector<T> &a) {
    for (auto &i: a) {
        std::cout << i << ' ';
    }
    std::cout << '\n';
}

int main() {
    collect_data_scl(7, 54, 15, 100000);
    return 0;
}