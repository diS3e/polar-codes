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

//Random
struct random rnd;

//Freezed subchannels
std::set<int> F;
std::vector<std::pair<double, int>> error_probability;
static double theta(double x) {
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

static double Q(double x) {
    return erfc(x / sqrt(2)) / 2;
}

double finding_E(int lambda, int i, double variability) {
    if (lambda == 0 && i == 0) {
        return 2 / variability;
    }

    if ((i & 1) == 0) {
        return theta(finding_E(lambda - 1, i >> 1, variability));
    } else {
        return 2 * finding_E(lambda - 1, i >> 1, variability);
    }
}

double finding_error_probability_awgn(int _m, int i, double variability) {
    return Q(sqrt(finding_E(_m, i, variability) / 2.0));
}
void find_freezed_channels(int m, int k, double variability) {
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
    std::vector<double> LLRs;
    std::vector<bool> activePath;
    std::vector<std::vector<double*>> arrayPointer_P;
    std::vector<std::vector<int*>> arrayPointer_C;
    std::vector<std::vector<int>> pathIndexToArrayIndex;
    std::vector<std::vector<int>> inactiveArrayIndices;
    std::vector<std::vector<int>> arrayReferenceCount;


    std::vector<int> reversed_permutation;

    SCL(int m, int k, int list_size): m(m), k(k), L(list_size) {

    }

    void _code(std::vector<int> &result, int l, int r) {
        if (r - l == 1) {
            return;
        }

        for (int i = l; i < l + ((r - l) >> 1); i++) {
            result[i] ^= result[((r - l) >> 1) + i];
        }
        _code(result, l, l + ((r - l) >> 1));
        _code(result, l + ((r - l) >> 1), r);
    }

    std::vector<int> code(std::vector<int> &information_word) {
        std::vector<int> result(1 << m, 0);
        for (int i = 0, j = 0; i < (1 << m); i++) {
            if (F.count(i) == 0) {
                result[i] = information_word[j];
                j++;
            }
        }
        _code(result, 0, 1 << m);
        return result;
    }

    void init_reversed_permutation() {
        reversed_permutation.resize(1 << m);
        for (int i = 0; i < (1 << m); ++i) {
            int res = 0;
            for (int j = 0; j < m; ++j) {
                res |= (((i & (1 << j)) == 0) ? 0 : 1) << (m - 1 - j);
            }
            reversed_permutation[i] = res;
        }
    }

    void initializeDataStructures() {
        inactivePathIndices.resize(L);
        inactiveArrayIndices.resize(m + 1);
        for (int i = 0; i < m + 1; ++i) {
            inactiveArrayIndices[i].resize(L);
        }
        LLRs.resize(L, 0);
        activePath.resize(L, false);
        arrayPointer_P.resize(m + 1, std::vector<double*>(L));
        arrayPointer_C.resize(m + 1, std::vector<int*>(L));
        pathIndexToArrayIndex.resize(m + 1, std::vector<int>(L, 0));
        arrayReferenceCount.resize(m + 1, std::vector<int>(L, 0));

        for (int lambda = 0; lambda < m + 1; ++lambda) {
            for (int s = 0; s < L; ++s) {
                arrayPointer_P[lambda][s] = new double[1 << (m - lambda)]();
                arrayPointer_C[lambda][s] = new int[2 * (1 << (m - lambda))]();
                for (int i = 0; i < 2 * (1 << (m - lambda)); ++i) {
                    arrayPointer_C[lambda][s][i] = -1;
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

    static int poptop(std::vector<int>& a) {
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

    double* getArrayPointer_P(int lambda, int l) {
        int s = pathIndexToArrayIndex[lambda][l];
        int s1;
        if (arrayReferenceCount[lambda][s] == 1) {
            s1 = s;
        } else {
            s1 = poptop(inactiveArrayIndices[lambda]);
            std::copy(arrayPointer_C[lambda][s], arrayPointer_C[lambda][s] + (1 << (m - lambda + 1)), arrayPointer_C[lambda][s1]);
            std::copy(arrayPointer_P[lambda][s], arrayPointer_P[lambda][s] + (1 << (m - lambda)), arrayPointer_P[lambda][s1]);
            arrayReferenceCount[lambda][s]--;
            arrayReferenceCount[lambda][s1] = 1;
            pathIndexToArrayIndex[lambda][l] = s1;
        }
        return arrayPointer_P[lambda][s1];
    }

    int* getArrayPointer_C(int lambda, int l) {
        int s = pathIndexToArrayIndex[lambda][l];
        int s1;
        if (arrayReferenceCount[lambda][s] == 1) {
            s1 = s;
        } else {

            s1 = poptop(inactiveArrayIndices[lambda]);

            std::copy(arrayPointer_C[lambda][s], arrayPointer_C[lambda][s] + (1 << (m - lambda + 1)), arrayPointer_C[lambda][s1]);
            std::copy(arrayPointer_P[lambda][s], arrayPointer_P[lambda][s] + (1 << (m - lambda)), arrayPointer_P[lambda][s1]);

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
            for (int beta = 0; beta < (1 << (m - lambda)); ++beta) {
                int left = 2 * beta;
                int right = 2 * beta + 1;
                if (phi % 2 == 0) {
                    auto a = P_lambda1[left];
                    auto b = P_lambda1[right];

                    P_lambda[beta] =
                            std::min(std::abs(a), std::abs(b)) *
                            (a > 0 ? 1.0 : -1.0) * (b > 0 ? 1.0 : -1.0);
                } else {
                    auto a = P_lambda1[left];
                    auto b = P_lambda1[right];
                    auto u1 = C_lambda[2 * beta];
                    P_lambda[beta] =
                            b + (1 - 2 * u1) * a;
                }
            }
        }
    }

    void recursivelyUpdateC(int lambda, int phi) {
        int psi = phi >> 1;
        for (int l = 0; l < L; ++l) {
            if (pathIndexInactive(l)) {
                continue;
            }
            auto C_lambda = getArrayPointer_C(lambda, l);
            auto C_lambda1 = getArrayPointer_C(lambda - 1, l);
            for (int beta = 0; beta < (1 << (m - lambda)); ++beta) {
                C_lambda1[2 * (2 * beta) + (psi % 2)] = C_lambda[2 * beta] ^ C_lambda[2 * beta + 1];
                C_lambda1[2 * (2 * beta + 1) + (psi % 2)] = C_lambda[2 * beta + 1];
            }
        }
        if (psi % 2 == 1) {
            recursivelyUpdateC(lambda - 1, psi);
        }
    }

    static int inline sign(double a) {
        if (a > 0) {
            return 1;
        } else if (a == 0) {
            std::terminate();
        } else {
            return -1;
        }
    }

    static double Phi(double mu, double lambda, int u) {
        if (2 * u == (1 - sign(lambda))) {
            return mu;
        } else {
            return mu + std::abs(lambda);
        }
    }


    void continuePaths_UnfrozenBit(int phi) {
        double probForks[L][2];
        int i = 0;
        for (int l = 0; l < L; ++l) {
            if (activePath[l]) {
                auto P_m = getArrayPointer_P(m, l);
                probForks[l][0] = Phi(LLRs[l], P_m[0], 0);
                probForks[l][1] = Phi(LLRs[l], P_m[0], 1);
                i++;
            } else {
                probForks[l][0] = std::numeric_limits<double>::max();
                probForks[l][1] = std::numeric_limits<double>::max();
            }
        }
        int rho = std::min(2 * i, L);


        bool contForks[L][2];
        for (int l = 0; l < L; ++l) {
            contForks[l][0] = false;
            contForks[l][1] = false;
        }
//        Отметить true там где максимальные rho элементов
        std::vector<std::pair<double, std::pair<int, int>>> find_max_array(L * 2);
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
                C_m[(phi % 2)] = 0;

                int l1 = clonePath(l);
                C_m = getArrayPointer_C(m, l1);
                C_m[phi % 2] = 1;
                auto P_m = getArrayPointer_P(m, l);
                LLRs[l] = Phi(LLRs[l], P_m[0], 0);
                P_m = getArrayPointer_P(m, l1);
                LLRs[l1] = Phi(LLRs[l1], P_m[0], 1);
            } else {
                if (contForks[l][0]) {
                    C_m[(phi % 2)] = 0;
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

    std::vector<int> decode(std::vector<double> &y) {
        initializeDataStructures();
        int l = assignInitialPath();
        auto P_0 = getArrayPointer_P(0, l);
        for (int beta = 0; beta < (1 << m); ++beta) {
            P_0[beta] = - y[reversed_permutation[beta]];
        }

        for (int phi = 0; phi < (1 << m); ++phi) {
            recursivelyCalcP(m, phi);
            if (F.count(phi) != 0) {
                for (int i = 0; i < L; ++i) {
                    if (!activePath[i]) {
                        continue;
                    }
                    auto C_m = getArrayPointer_C(m, i);
                    C_m[phi % 2] = 0;
                    auto P_m = getArrayPointer_P(m, i);
                    LLRs[i] = Phi(LLRs[i], P_m[0], 0);
                }
            } else {
                continuePaths_UnfrozenBit(phi);
            }
            if (phi % 2 == 1) {
                recursivelyUpdateC(m, phi);
            }

        }

        int l1 = 0;
        double p1 = std::numeric_limits<double>::max();
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
            res[reversed_permutation[i]] = C_0[2 * i];
        }
        for (int lambda = 0; lambda < m + 1; ++lambda) {
            for (int i = 0; i < L; ++i) {
                delete[] arrayPointer_C[lambda][i];
                delete[] arrayPointer_P[lambda][i];
            }
        }
        return res;
    }

    std::vector<double> get_corrupt(std::vector<int> &random_codeword, double arg) const {
        int n = 1 << m;
        std::vector<double> ans(n, 0);
        for (int i = 0; i < n; ++i) {
            ans[i] = ((random_codeword[i] * 2) - 1) +
                     sqrt(static_cast<double>(n) / (2 * arg * (n - k))) * rnd.normal(rnd.rng);
        }
        return ans;
    }
};

bool words_equals(std::vector<int> const &a, std::vector<int> const &b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (int i = 0; i < a.size(); ++i) {
        if (!((a[i] == 0 || a[i] == 1) && (b[i] == 0 || b[i] == 1))) {
            std::terminate();
        }
        if (a[i] != b[i]) return false;
    }
    return true;
}

void collect_data_scl(int m, int k, int L, int precision) {
    for (double Eb = 1.5; Eb < 5.5; Eb += 0.5) {

        double variability = static_cast<double>((1 << m)) / (2 * pow(10.0, (Eb / 10)) * ((1 << m) - k));
        find_freezed_channels(m, k, variability);
        volatile double correct = 0;

        double all = precision;
//#pragma omp parallel for reduction(+:correct, all)
        for (int tries = 0; tries < precision; tries++) {
            SCL scl(m, k, L);
            scl.init_reversed_permutation();

            auto information_word = rnd.get_random_word((1 << scl.m) - scl.k);
            auto coded_word = scl.code(information_word);
            std::vector<double> corrupt = scl.get_corrupt(coded_word, pow(10.0, (Eb / 10)));
            auto result = scl.decode(corrupt);
            scl._code(result, 0, 1 << scl.m);
            std::vector<int> result_without_freezing;
            for (int i = 0; i < result.size(); ++i) {
                if (F.count(i) == 0) {
                    result_without_freezing.push_back(result[i]);
                }
            }

            if (words_equals(result_without_freezing, information_word)) {
                correct++;
            }
        }
        std::cout << std::fixed << std::setprecision(6) << Eb << ";" << (all - correct) / all << '\n';
    }
}

int main() {
    collect_data_scl(6, 28, 4, 10000);
    return 0;
}