#include <bits/stdc++.h>
using namespace std;

const double INF = numeric_limits<double>::max();

double Euclidean (const vector<double> &a, const vector<double> &b) {
    double res = 0;
    const int n = (int)a.size();
    for (int i = 0; i < n; ++i) {
        res += (a[i] - b[i]) * (a[i] - b[i]);  // 計算平方差並累加
    }
    return sqrt(res); // 返回歐幾里得距離
}

vector<int> encode (const vector<vector<double>> &data, const vector<vector<double>> &centroids) {
    vector<int> res;
    const int n = (int)centroids.size();
    for (auto& v : data) {
        double min_dist = INF; // 初始化最小距離為無限大
        int id = -1; // 初始化最近的 centroid 索引為 -1
        for (int i = 0; i < n; ++i) { // 計算向量 v 與第 i 個 centroid 之間的歐幾里德距離
            double dist = Euclidean(v, centroids[i]);
            if (dist < min_dist) { // 更新距離和 centroid id
                min_dist = dist;
                id = i;
            }
        }
        res.emplace_back(id);  // 最近的 centroid 索引加入結果向量 res
    }
    return res;
}

vector<vector<double>> decode (const vector<int> &data, const vector<vector<double>> &centroids) {
    vector<vector<double>> res;
    const int n = (int)centroids.size();
    for (int i: data) {
        if (i >= 0 and i < n) res.emplace_back(centroids[i]);  // 索引 i 有效，則將對應的 centroid 向量加入結果向量 res
        else res.emplace_back(vector<double>()); // 索引 i 無效，則加入一個空的向量
    }
    return res;
}

// 設定 random seed
auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
mt19937 Generate(seed);

const int MX = 1e9; // random max
int main (void) {
    int dim = 3, n = 256, input = 10;
    uniform_int_distribution<int> random(1, MX);
    vector<vector<double>> data;

    for (int i = 0; i < n; ++i) {
        vector<double> v;
        for (int j = 0; j < dim; ++j) v.emplace_back(1.0 * random(Generate) / MX); // 產生 dim 個介於 0 到 1 之間的隨機 double 值，並加入向量 v
        data.emplace_back(v);
    }

    vector<vector<double>> v;
    for (int i = 0; i < input; ++i) {
        vector<double> tmp;
        for (int j = 0; j < dim; ++j) tmp.emplace_back(1.0 * random(Generate) / MX);  // 產生 dim 個介於 0 到 1 之間的隨機 double 值，並加入向量 tmp
        v.emplace_back(tmp);
    }

    vector<int> encoded = encode(v, data); // 使用 encode 函式將輸入向量 v 編碼，centroid 為 data
    cout << "Encoded: " << '\n';
    for (int i: encoded) cout << i << ' ';
    cout << "\n\n";

    vector<vector<double>> decoded = decode(encoded, data); // 使用 decode 函式將編碼結果解碼，centroid 為 data
    cout << "Decoded: " << '\n';
    for (auto &v: decoded) {
        for (double d: v) cout << d << ' ';
        cout << '\n';
    }
    return 0;
}