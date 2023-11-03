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
int k = 28;

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

constexpr int NOT_EVALUATED = -239239;

void update_u_memory(int lambda, int j, std::vector<std::vector<int>> &u_memory) {
    if (lambda == 0) {
        return;
    }
    int i = j >> 1;
    int left = (2 * i) - (i % (1 << (lambda - 1)));
    int right = (1 << (lambda - 1)) + (2 * i) - (i % (1 << (lambda - 1)));
    if (j % 2 == 1) {
        u_memory[lambda - 1][left] = u_memory[lambda][2 * i] ^ u_memory[lambda][2 * i + 1];
        u_memory[lambda - 1][right] = u_memory[lambda][2 * i + 1];
        update_u_memory(lambda - 1, left, u_memory);
        update_u_memory(lambda - 1, right, u_memory);
    }

}

std::vector<int> reversed_permutation(1 << m);


void S(int lambda, int j, std::vector<std::vector<int>> &u_memory, std::vector<std::vector<double>> &L_memory) {

    if (lambda == 0) {
        return;
    }

    int i = j >> 1;

    int left = (2 * i) - (i % (1 << (lambda - 1)));
    int right = (1 << (lambda - 1)) + (2 * i) - (i % (1 << (lambda - 1)));

    double a;

    if (L_memory[lambda - 1][left] == NOT_EVALUATED) {
        S(lambda - 1, left,  u_memory, L_memory);
    }
    a = L_memory[lambda - 1][left];
    double b;
    if (L_memory[lambda - 1][right] == NOT_EVALUATED) {
        S(lambda - 1, right, u_memory, L_memory);
    }
    b = L_memory[lambda - 1][right];

    double res;
    if (j % 2 == 0) {
        res = std::min(std::abs(a), std::abs(b)) * (a > 0 ? 1.0 : -1.0) * (b > 0 ? 1.0 : -1.0);
    } else {
        res = b + (((u_memory[lambda][2 * i] % 2 == 0) ? 1.0 : -1.0) * a);
    }
    L_memory[lambda][j] = res;
}


std::vector<int> decode(std::vector<double> &corrupted_word) {
    std::vector<int> result;
    std::vector<std::vector<int>> u_memory(m + 1, std::vector<int>((1 << m), NOT_EVALUATED));
    std::vector<std::vector<double>> L_memory(m + 1, std::vector<double>(1 << m, NOT_EVALUATED));
    for (int i = 0; i < (1 << m); ++i) {
        L_memory[0][i] = -corrupted_word[reversed_permutation[i]];
    }
    for (int i = 0; i < (1 << m); i++) {
        if (F.count(i) == 0) {
            S(m, i, u_memory, L_memory);
            int res = (L_memory[m][i] > 0) ? 0 : 1;
            result.push_back(res);
            u_memory[m][i] = res;
        } else {
            result.push_back(0);
            u_memory[m][i] = 0;
        }
        update_u_memory(m, i, u_memory);
    }
    return result;
}

int inline sign(double a) {
    if (a > 0) {
        return 1;
    } else if (a == 0) {
      return 0;
    } else {
        return -1;
    }
}

double phi(double mu, double lambda, int u) {
    if (2 * u == (1 - sign(lambda))) {
        return mu;
    } else {
        return mu + std::abs(lambda);
    }
}

struct path {
    double LLR = 0;
    std::vector<int> word{};
    int memory_link = 0;
};

std::vector<int> list_decoder(std::vector<double> &corrupted_word, int L) {
    std::vector<std::vector<std::vector<double>>> path_L_memory(L, std::vector<std::vector<double>>(m + 1, std::vector<double>(1 << m, NOT_EVALUATED)));
    std::vector<std::vector<std::vector<int>>> path_u_memory(L, std::vector<std::vector<int>>(m + 1, std::vector<int>((1 << m), NOT_EVALUATED)));
    for (int j = 0; j < L; ++j) {
        for (int i = 0; i < (1 << m); ++i) {
            path_L_memory[j][0][i] = -corrupted_word[reversed_permutation[i]];
        }
    }
    auto cmp = [](path &left, path &right){
        if (left.word.size() < right.word.size()) {
            return false;
        } else if (left.word.size() > right.word.size()) {
            return true;
        }
        return left.LLR > right.LLR;
    };

    std::priority_queue<path, std::vector<path>, decltype(cmp)> list(cmp);
    list.emplace();
    for (int i = 0; i < (1 << m); ++i) {
        if (F.count(i) != 0) {
            auto size = list.size();
            for (int j = 0; j < size; ++j) {
                auto top_element = list.top();
                list.pop();
                top_element.word.push_back(0);
                S(m, i, path_u_memory[top_element.memory_link], path_L_memory[top_element.memory_link]);
                top_element.LLR = phi(top_element.LLR, path_L_memory[top_element.memory_link][m][i], 0);
                path_u_memory[top_element.memory_link][m][i] = 0;
                update_u_memory(m, i, path_u_memory[top_element.memory_link]);
                list.push(top_element);

            }
        } else {
            std::priority_queue<path, std::vector<path>, decltype(cmp)> new_list(cmp);
            auto size = list.size();
            for (int j = 0; j < size; ++j) {
                auto zero_next = list.top();
                list.pop();
                auto one_next = zero_next;
                zero_next.word.push_back(0);
                one_next.word.push_back(1);
                S(m, i, path_u_memory[zero_next.memory_link], path_L_memory[zero_next.memory_link]);

                zero_next.LLR = phi(zero_next.LLR, path_L_memory[zero_next.memory_link][m][i], 0);
                one_next.LLR = phi(one_next.LLR, path_L_memory[one_next.memory_link][m][i], 1);

                new_list.push(zero_next);
                new_list.push(one_next);

            }

            std::vector<std::vector<std::vector<double>>> new_path_L_memory(L, std::vector<std::vector<double>>(m + 1));
            std::vector<std::vector<std::vector<int>>> new_path_u_memory(L, std::vector<std::vector<int>>(m + 1));
            for (int j = 0; j < L && !new_list.empty(); ++j) {
                auto top = new_list.top();
                new_path_L_memory[j] = (path_L_memory[top.memory_link]);
                new_path_u_memory[j] = (path_u_memory[top.memory_link]);
                new_path_u_memory[j][m][i] = top.word[top.word.size() - 1];
                update_u_memory(m, i, new_path_u_memory[j]);
                top.memory_link = j;
                list.push(top);
                new_list.pop();
            }
            path_L_memory = new_path_L_memory;
            path_u_memory = new_path_u_memory;
        }
    }

    auto result = list.top().word;
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

constexpr int ITERATION = 100000;

void collect_data(int precision) {
    for (double Eb = 0; Eb < 5.5; Eb += 0.5) {
        double variability = static_cast<double>((1 << m)) / (2 * pow(10.0, (Eb / 10)) * ((1 << m) - k));
        find_freezed_channels(variability);

        volatile double correct = 0;
        double all = precision;
        #pragma omp parallel for reduction(+:correct, all)
        for (int tries = 0; tries < ITERATION; tries++) {
            auto information_word = get_random_word((1 << m) - k);
            auto coded_word = code(information_word);
            std::vector<double> corrupt = get_corrupt(coded_word, pow(10.0, (Eb / 10)));
            auto result = decode(corrupt);
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

void init_reversed_permutation() {
    for (int i = 0; i < (1 << m); ++i) {
        int res = 0;
        for (int j = 0; j < m; ++j) {
            res |= (((i & (1 << j)) == 0) ? 0 : 1) << (m - 1 - j);
        }
        reversed_permutation[i] = res;
    }
}

int main() {
    init_reversed_permutation();
    collect_data(ITERATION);
}