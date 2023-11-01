#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <set>
#include <string>
#include <random>
#include <chrono>
#include <iomanip>
#include <queue>


void printVector(const std::vector<int> &vector, const std::string &title) {
    std::cout << title << '\n';
    for (auto t: vector) {
        std::cout << t << ' ';
    }
    std::cout << '\n';
}

std::vector<int> odd(const std::vector<int> &vector, int max_index) {
    std::vector<int> result;
    for (int i = 0; (i << 1) + 1 < max_index; i++) {
        result.push_back(vector[(i << 1) + 1]);
    }
    return result;
}

std::vector<int> even(const std::vector<int> &vector, int max_index) {
    std::vector<int> result;
    for (int i = 0; (i << 1) < max_index; i++) {
        result.push_back(vector[i << 1]);
    }
    return result;
}


std::vector<double> odd(const std::vector<double> &vector, int max_index) {
    std::vector<double> result;
    for (int i = 0; (i << 1) + 1 < max_index; i++) {
        result.push_back(vector[(i << 1) + 1]);
    }
    return result;
}

std::vector<double> even(const std::vector<double> &vector, int max_index) {
    std::vector<double> result;
    for (int i = 0; (i << 1) < max_index; i++) {
        result.push_back(vector[i << 1]);
    }
    return result;
}


//Code parameters
int m = 6;
int k = 34;

//AWGN parameters

//Binary erase parameters
double p = 0.5;

//Freezed subchannels
std::set<int> F;
std::vector<std::pair<double, int>> error_probability;

//AWGN channel
double theta(double x) {
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

double Q(double x) {
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

double finding_error_probability_awgn(int m, int i, double variability) {
    return Q(sqrt(finding_E(m, i, variability) / 2.0));
}

//Binary erase channel
double finding_error_probability_erase(int m, int i) {
    if (m == 0 && i == 0) {
        return p;
    }

    if ((i & 1) == 0) {
        double z_prev = finding_error_probability_erase(m - 1, i >> 1);
        return 2 * z_prev - z_prev * z_prev;
    } else {
        double z_prev = finding_error_probability_erase(m - 1, i >> 1);
        return z_prev * z_prev;
    }
}

void find_freezed_channels(double variability) {

    for (int i = 0; i < (1 << m); ++i) {
        error_probability.emplace_back(finding_error_probability_awgn(m, i, variability), i);
    }

    auto comparator = [](const auto &a, const auto &b) { return a.first > b.first; };
    std::sort(error_probability.begin(), error_probability.end(), comparator);

    for (int i = 0; i < k; ++i) {
        F.insert(error_probability[i].second);
    }
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


double S(int lambda, int j, std::vector<int> &u, std::vector<double> &y) {
    if (j == 0 && lambda == 0) {
        return -y[0];
    }

    int i = j >> 1;

    auto v_e = even(u, 2 * i);
    auto v_o = odd(u, 2 * i);
    auto y_e = even(y, y.size());
    auto y_o = odd(y, y.size());

    std::vector<int> xorr(v_e);
    for (int l = 0; l < xorr.size(); ++l) {
        xorr[l] ^= v_o[l];
    }

    double a = S(lambda - 1, i, xorr, y_e);
    double b = S(lambda - 1, i, v_o, y_o);
    if (j % 2 == 0) {
        return std::min(std::abs(a), std::abs(b)) * (a > 0 ? 1.0 : -1.0) * (b > 0 ? 1.0 : -1.0);
    } else {
        return b + (((u[2 * i] % 2 == 0) ? 1.0 : -1.0) * a);
    }
}


std::vector<int> decode(std::vector<double> &corrupted_word) {
    std::vector<int> result;

    for (int i = 0; i < (1 << m); i++) {
        if (F.count(i) == 0) {
            double res = S(m, i, result, corrupted_word);
            result.push_back((res > 0) ? 0 : 1);
        } else {
            result.push_back(0);
        }
    }
    return result;
}


std::vector<int> list_decoder(std::vector<double> &corrupted_word, int L) {
    std::priority_queue<std::pair<double, std::vector<int>>> list;
    list.push({0, {}});
//    list.push_back(std::make_pair(0, std::vector<int>(0)))
    for (int i = 0; i < (1 << m); ++i) {
        std::priority_queue<std::pair<double, std::vector<int>>> new_list;
        std::vector<std::vector<int>> new_layer;
        if (F.count(i) != 0) {
            auto size = list.size();
            for (int j = 0; j < size; ++j) {
                auto top_element = list.top().second;
                list.pop();
                top_element.push_back(0);
                new_layer.push_back(top_element);
            }
        } else {
            auto size = list.size();
            for (int j = 0; j < size; ++j) {
                auto top_element = list.top().second;
                list.pop();
//                double res = S(m, i, top_element, corrupted_word);
//                top_element.push_back((res > 0) ? 0 : 1);
//                top_element.push_back(0);
                auto copy_top_element = top_element;
                copy_top_element.push_back(1);
                top_element.push_back(0);
                new_layer.push_back(top_element);
                new_layer.push_back(copy_top_element);
            }
        }
        for (int j = 0; j < new_layer.size(); ++j) {
            double lop = S(m, i, new_layer[j], corrupted_word);
            new_list.emplace(lop * (-2 * new_layer[j][new_layer[j].size() - 1] + 1), new_layer[j]);
        }
        while(!list.empty()) {
            list.pop();
        }
        for (int j = 0; j < L; ++j) {
            auto top_elelment = new_list.top();
            list.push(top_elelment);
        }
        
    }

    auto result = list.top().second;
//    for (int i = 0; i < (1 << m); i++) {
//        if (F.count((1 << m) - 1) == 0) {
//            double res = S(m, (1 << m) - 1, result, corrupted_word);
//            result.push_back((res > 0) ? 0 : 1);
//        } else {
//            result.push_back(0);
//        }
//    }
return result;
}


//Random
std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
std::uniform_int_distribution<int> uniform_int(0, std::numeric_limits<int>::max());
std::uniform_real_distribution<> uniform_real(0.0, 1.0);
std::normal_distribution<double> normal(0, 1);

int rnd(int l, int r) {
    return (uniform_int(rng) % (r - l + 1) + l);
}

std::vector<int> get_random_word(int length) {
    std::vector<int> result(length);
    for (auto &t: result) t = rnd(0, 1);
    return result;
}

std::vector<double> get_corrupt(std::vector<int> &random_codeword, double arg) {
    int n = 1 << m;
    std::vector<double> ans(n, 0);
    for (int i = 0; i < n; ++i) {
        ans[i] = ((random_codeword[i] * 2) - 1) +
                 sqrt(static_cast<double>(n) / (2 * arg * (n - k))) * normal(rng);
    }
    return ans;
}

std::vector<int> get_signs(std::vector<double> &corrupted) {
    std::vector<int> ans(corrupted.size());
    for (int i = 0; i < corrupted.size(); ++i) {
        ans[i] = (corrupted[i] > 0) ? 1 : 0;
    }
    return ans;
}

std::vector<double> get_reliability(std::vector<double> &corrupted) {
    std::vector<double> ans(corrupted.size());
    for (int i = 0; i < ans.size(); ++i) {
        ans[i] = std::abs(corrupted[i]);
    }
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

void collect_data(int precision) {
    for (double Eb = 3.0; Eb < 5.5; Eb += 0.5) {
        double variability = static_cast<double>((1 << m)) / (2 * pow(10.0, (Eb / 10)) * ((1 << m) - k));
        find_freezed_channels(variability);

        double correct = 0;
        double all = precision;
        for (int tries = 0; tries < all; tries++) {
            auto information_word = get_random_word((1 << m) - k);
            auto coded_word = code(information_word);
            std::vector<double> corrupt = get_corrupt(coded_word, pow(10.0, (Eb / 10)));
//            auto result = list_decoder(corrupt, 256);
            auto result = decode(corrupt);
//            auto list_result = list_decoder(corrupt, 1);
            std::vector<int> result_without_freezing;
            for (int i = 0; i < result.size(); ++i) {
                if (F.count(i) == 0) {
                    result_without_freezing.push_back(result[i]);
                }
            }

            if (words_equals(result_without_freezing, information_word)) {
                correct++;
//                std::vector<int> result_list_without_freezing;
//                for (int i = 0; i < list_result.size(); ++i) {
//                    if (F.count(i) == 0) {
//                        result_list_without_freezing.push_back(list_result[i]);
//                    }
//                }
//                if (!words_equals(result_list_without_freezing, information_word)) {
//                    int x = 1;
//                    x += 1;
//                }
            }
        }
        std::cout << std::fixed << std::setprecision(6) << Eb << ";" << (all - correct) / all << '\n';
    }
}

int main() {
    collect_data(1000);
//    std::priority_queue<std::pair<double, std::vector<int>>> priorityQueue;
//    priorityQueue.push({1.1, {1, 0}});
//    priorityQueue.push({1.2, {1, 0, 1}});
//    priorityQueue.push({1.3, {1, 1}});
//    priorityQueue.push({1.0, {0}});
//    priorityQueue.push({1.0, {0, 1}});
//    std::cout << priorityQueue.size() << '\n';
//    int size = priorityQueue.size();
//    for (int i = 0; i < size; ++i) {
//        std::cout << priorityQueue.top().first << ' ' << priorityQueue.top().second.size() << '\n';
//        priorityQueue.pop();
//    }
//    find_freezed_channels(5.0);
////    std::vector<int> info_word{1, 1, 1, 1, 1};
////    auto coded = code(info_word);
//    std::vector<double> corrupted_word{-0.10039712500291453, 0.12456834960559582, 0.031906924486806432, -1.7838410749533051, -1.4241922215541076, -1.5579572196246154, 1.7401530350590679, -0.45626271517822514};
//
////    auto
//    auto res = list_decoder(corrupted_word, 1);
//    printVector(res, "List decoded: ");
    return 0;

}

// (,) (f a) (f b) -> f ((,) a b)