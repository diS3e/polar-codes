#include "PolarCode.h"
#include <vector>
#include "stack.h"
#include "ChannelAWGN.h"
#include <immintrin.h>

PolarCode::PolarCode(int m, int k, int list_size, float variability) :
        m(m),
        k(k),
        L(list_size),
        inactivePathIndices(L),
        inactiveArrayIndices(m + 1, stack<int>(L)),
        LLRs(L),
        activePath(L),
        arrayPointer_P(m + 1, std::vector<float *>(L)),
        arrayPointer_C(m + 1, std::vector<int *>(L)),
        pathIndexToArrayIndex(m + 1, std::vector<int>(L)),
        arrayReferenceCount(m + 1, std::vector<int>(L)),
        polarizedChannel(m, k, variability) {
    find_max_array = (std::pair<float, int> *) malloc(sizeof(std::pair<float, int>) * L * 2);
    contForks = (bool *) malloc(sizeof(bool) * L * 2);
    for (int lambda = 0; lambda < m + 1; ++lambda) {
        for (int s = 0; s < L; ++s) {
            arrayPointer_P[lambda][s] = new float[1 << (m - lambda)]();
            arrayPointer_C[lambda][s] = new int[(1 << (m - lambda + 1))]();
        }
    }

}

void PolarCode::initializeDataStructures() {
    for (int i = 0; i < m + 1; ++i) {
        inactiveArrayIndices[i].clear();
    }

    std::fill(activePath.begin(), activePath.end(), false);
    for (int lambda = 0; lambda < m + 1; ++lambda) {
        std::fill(arrayReferenceCount[lambda].begin(), arrayReferenceCount[lambda].end(), 0);
        for (int s = 0; s < L; ++s) {
#ifdef DEBUG_MODE
            for (int i = 0; i < (1 << (m - lambda + 1)); ++i) {
                arrayPointer_C[lambda][s][i] = 239;
            }
#endif
            inactiveArrayIndices[lambda].push(L - 1 - s);
            pathIndexToArrayIndex[lambda][s] = -1;
        }
    }
    for (int l = L - 1; l >= 0; --l) {
        inactivePathIndices.push(l);
    }
}

PolarCode::~PolarCode() {
    free(contForks);
    for (int lambda = 0; lambda < m + 1; ++lambda) {
        for (int i = 0; i < L; ++i) {
            delete[] arrayPointer_C[lambda][i];
            delete[] arrayPointer_P[lambda][i];
        }
    }
}

void PolarCode::_encode(std::vector<int> &result, int l, int r) {
    if (r - l == 1) {
        return;
    }

    for (int i = l; i < l + ((r - l) >> 1); i++) {
        result[i] ^= result[((r - l) >> 1) + i];
    }
    _encode(result, l, l + ((r - l) >> 1));
    _encode(result, l + ((r - l) >> 1), r);
}

void PolarCode::encode_inplace(const std::vector<int> &information_word, std::vector<int> &result) {
    for (int i = 0, j = 0; i < (1 << m); i++) {
        if (!polarizedChannel.isFrozen(i)) {
            result[i] = information_word[j];
            j++;
        } else {
            result[i] = 0;
        }
    }
    _encode(result, 0, 1 << m);
}

std::vector<int> PolarCode::encode(std::vector<int> &information_word) {
    std::vector<int> result(1 << m, 0);
    encode_inplace(information_word, result);
    return result;
}

int PolarCode::assignInitialPath() {
    int l = inactivePathIndices.pop();
    activePath[l] = true;
    for (int lambda = 0; lambda < m + 1; ++lambda) {
        int s = inactiveArrayIndices[lambda].pop();
        pathIndexToArrayIndex[lambda][l] = s;
        arrayReferenceCount[lambda][s] = 1;
    }
    return l;
}

int PolarCode::clonePath(int l) {
    int l1 = inactivePathIndices.pop();
    activePath[l1] = true;
    LLRs[l1] = LLRs[l];
    for (int lambda = 0; lambda < m + 1; ++lambda) {
        int s = pathIndexToArrayIndex[lambda][l];
        pathIndexToArrayIndex[lambda][l1] = s;
        arrayReferenceCount[lambda][s]++;
    }
    return l1;
}

void PolarCode::killPath(int l) {
    activePath[l] = false;
    inactivePathIndices.push(l);
    LLRs[l] = 0;
    for (int lambda = 0; lambda < m + 1; ++lambda) {
        int s = pathIndexToArrayIndex[lambda][l];
        arrayReferenceCount[lambda][s]--;
        if (arrayReferenceCount[lambda][s] == 0) {
            inactiveArrayIndices[lambda].push(s);
        }
    }
}

float *PolarCode::getArrayPointer_P(int lambda, int l) {
    int s = pathIndexToArrayIndex[lambda][l];
    int s1;
    if (arrayReferenceCount[lambda][s] == 1) {
        s1 = s;
    } else {
        s1 = inactiveArrayIndices[lambda].pop();
        arrayReferenceCount[lambda][s]--;
        arrayReferenceCount[lambda][s1] = 1;
        pathIndexToArrayIndex[lambda][l] = s1;
    }
    return arrayPointer_P[lambda][s1];
}

int *PolarCode::getArrayPointer_C(int lambda, int l) {
    int s = pathIndexToArrayIndex[lambda][l];
    return arrayPointer_C[lambda][s];
}

bool PolarCode::pathIndexInactive(int l) {
    if (activePath[l]) {
        return false;
    } else {
        return true;
    }
}


__m256 PolarCode::copy_sign(__m256 srcSign, __m256 srcValue) {
    const __m256 mask0 = _mm256_set1_ps(-0.);
    __m256 tmp0 = _mm256_and_ps(srcSign, mask0);
    __m256 tmp1 = _mm256_andnot_ps(mask0, srcValue);
    return _mm256_or_ps(tmp0, tmp1);
}

__m256 PolarCode::f_avx(float *pA, float *pB) {
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

__m256 PolarCode::g_avx(float *pA, float *pB, int *pC) {
    __m256 vA = _mm256_loadu_ps(pA);
    __m256 vB = _mm256_loadu_ps(pB);
    __m256 vC = _mm256_loadu_ps((float *) pC);
    __m256 signs = copy_sign(vC, _mm256_set1_ps(1.f));
    return _mm256_sub_ps(vB, _mm256_mul_ps(signs, vA));
}


void PolarCode::recursivelyCalcP(int lambda, int phi) {
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
        auto P_lambda1 = arrayPointer_P[lambda - 1][pathIndexToArrayIndex[lambda - 1][l]];
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

int PolarCode::countPhaseIndex(int phase) {
    int index = 0;
    for (int i = 1; i < (1 << m); i <<= 1, index++) {
        if ((i & phase) == 0) {
            return index;
        }
    }
    return index;
}

void PolarCode::recursivelyUpdateC(int phi) {
    for (int l = 0; l < L; ++l) {
        if (pathIndexInactive(l)) {
            continue;
        }
        int put_pointer = 0;
        int index = countPhaseIndex(phi);
        int s_index = pathIndexToArrayIndex[m - index][l];

        if (arrayReferenceCount[m - index][s_index] != 1) {
            int new_inactive_array = inactiveArrayIndices[m - index].pop();
            pathIndexToArrayIndex[m - index][l] = new_inactive_array;
            arrayReferenceCount[m - index][s_index]--;
            arrayReferenceCount[m - index][new_inactive_array] = 1;
            s_index = new_inactive_array;
        }

        for (int i = 0; i < index; ++i) {
            for (int j = 0; j < (1 << i); ++j) {
                int s = pathIndexToArrayIndex[m - i][l];
                arrayPointer_C[m - index][s_index][(1 << index) - (1 << (i + 1)) + j] = arrayPointer_C[m - i][s][j];
                put_pointer++;
            }
        }

        arrayPointer_C[m - index][s_index][put_pointer] = arrayPointer_C[m - 0][pathIndexToArrayIndex[m - 0][l]][1];
        for (int i = 0; i < index; ++i) {
            for (int j = 0; j < (1 << i); ++j) {
                arrayPointer_C[m - index][s_index][(1 << index) - (1 << (i + 1)) + j]
                        *= -arrayPointer_C[m - index][s_index][(1 << index) - (1 << i) + j];
            }
        }
    }

}

int inline PolarCode::sign(float a) {
    if (a > 0) {
        return 1;
    } else {
        return -1;
    }
}

float PolarCode::Phi(float mu, float lambda, int u) {
    if (2 * u == (1 - sign(lambda))) {
        return mu;
    } else {
        return mu + std::abs(lambda);
    }
}

void PolarCode::continuePaths_UnfrozenBit(int phi) {
    int i = 0;
    for (int l = 0; l < L; ++l) {
        if (activePath[l]) {
            auto P_m = getArrayPointer_P(m, l);
            find_max_array[2 * l] = {Phi(LLRs[l], P_m[0], 0), 2 * l};
            find_max_array[2 * l + 1] = {Phi(LLRs[l], P_m[0], 1), 2 * l + 1};

            i++;
        } else {
            find_max_array[2 * l] = {std::numeric_limits<float>::max(), 2 * l};
            find_max_array[2 * l + 1] = {std::numeric_limits<float>::max(), 2 * l + 1};
        }
    }
    int rho = std::min(2 * i, L);

    for (int l = 0; l < L * 2; ++l) {
        contForks[l] = false;
    }

    std::nth_element(find_max_array, find_max_array + rho - 1, find_max_array + L * 2);
    for (int j = 0; j < rho; ++j) {
        auto reliable_path = find_max_array[j].second;
        contForks[reliable_path] = true;
    }
    for (int l = 0; l < L; ++l) {
        if (pathIndexInactive(l)) {
            continue;
        }
        if (!contForks[2 * l] && !contForks[2 * l + 1]) {
            killPath(l);
        }
    }
    for (int l = 0; l < L; ++l) {
        if (!contForks[2 * l] && !contForks[2 * l + 1]) {
            continue;
        }
        auto C_m = getArrayPointer_C(m, l);
        if (contForks[2 * l] && contForks[2 * l + 1]) {
            C_m[(phi % 2)] = -1;

            int l1 = clonePath(l);
            int s = pathIndexToArrayIndex[m][l1];
            int s1 = inactiveArrayIndices[m].pop();
            // copy trees root
            arrayPointer_C[m][s1][0] = arrayPointer_C[m][s][0];
            arrayPointer_C[m][s1][1] = arrayPointer_C[m][s][1];
            arrayPointer_P[m][s1][0] = arrayPointer_P[m][s][0];
            arrayReferenceCount[m][s]--;
            arrayReferenceCount[m][s1] = 1;
            pathIndexToArrayIndex[m][l] = s1;
            C_m = getArrayPointer_C(m, l1);
            C_m[phi % 2] = 1;
            auto P_m = getArrayPointer_P(m, l);
            LLRs[l] = Phi(LLRs[l], P_m[0], 0);
            P_m = getArrayPointer_P(m, l1);
            LLRs[l1] = Phi(LLRs[l1], P_m[0], 1);
        } else {
            if (contForks[2 * l]) {
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

std::vector<int> PolarCode::decode(std::vector<float> &y) {
    initializeDataStructures();
    int l = assignInitialPath();
    auto P_0 = getArrayPointer_P(0, l);
    for (int beta = 0; beta < (1 << m); ++beta) {
        P_0[beta] = -y[beta];
    }
    for (int phi = 0; phi < (1 << m); ++phi) {
        recursivelyCalcP(m, phi);
        if (polarizedChannel.isFrozen(phi)) {
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
            recursivelyUpdateC(phi);
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
                    if (arrayPointer_C[lambda][s][j] == 239) {
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
    return res;
}
