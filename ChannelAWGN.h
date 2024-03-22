#ifndef POLAR_CODES_CHANNELAWGN_H
#define POLAR_CODES_CHANNELAWGN_H

#include <set>
#include <vector>
#include <cmath>
#include <algorithm>

class ChannelAWGN {
public:
    ChannelAWGN(int m, int k, float variability) {
        find_freezed_channels(m, k, variability);
    }

    bool isFrozen(int i) {
        return F.count(i);
    }
private:
    std::set<int> F;
    std::vector<std::pair<float, int>> error_probability;
    void find_freezed_channels(int m, int k, float variability);
};


#endif //POLAR_CODES_CHANNELAWGN_H
