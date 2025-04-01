#include <bits/stdc++.h>
using namespace std;

using ll = long long;
const ll mod = INT_MAX;

int n = 5, m = 5;
using pi = pair<long, long>;
// 快速冪
ll power (ll a, ll b) { ll ans = 1; for (; b; b >>= 1, a = a * a % mod) if (b & 1) ans = ans * a % mod; return ans; }
ll inv (ll a) { return power(a, mod - 2); } // modulo inverse

auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
mt19937 Generate(seed);

// 編碼函數：將秘密值 secret 分割成 n 個份額
vector<pi> encode(ll secret) {
    vector<ll> coe(m);                                 // 儲存多項式的係數，大小為 m
    coe[0] = secret;                                   // 多項式的常數項 (coe[0]) 設為秘密值
    uniform_int_distribution<ll> distrib(1, mod - 1);  // 定義一個均勻分布，用於生成 1 到 mod-1 之間的隨機數

    // 隨機生成多項式的其他係數 (coe[1] 到 coe[m-1])
    for (int i = 1; i < m; ++i) {
        coe[i] = distrib(Generate);
    }

    vector<pi> ret(n);  // 儲存產生的 n 個份額

    // 生成 n 個份額 (點)
    for (int i = 0; i < n; ++i) {
        ll x = i + 1;  // x 座標從 1 開始取
        ll y = 0;      // 初始化 y 座標
        ll p = 1;      // 用於計算 x 的冪次 (x^0, x^1, x^2, ...)

        // 計算多項式在 x 點的值 y = P(x) = coe[0]*x^0 + coe[1]*x^1 + ... + coe[m-1]*x^(m-1)
        for (int j = 0; j < m; ++j) {
            ll term = coe[j] * p % mod;  // 計算每一項 coe[j] * x^j
            y = (y + term) % mod;        // 將項加到 y 並取模
            p = p * x % mod;             // 更新 x 的冪次 (p = x^(j+1))
        }
        ret[i] = {x, y};  // 將產生的點 (x, y) 存入結果 vector
    }
    return ret;  // 回傳 n 個份額
}

// 解碼函數：使用 m 個份額 v 來還原秘密值
ll decode(const vector<pi>& v) {
    ll secret = 0;  // 初始化還原的秘密值

    // 使用拉格朗日插值法找出多項式的常數項 (即秘密值)
    // P(0) = sum( y_i * L_i(0) ) for i = 0 to m-1
    // L_i(x) = product( (x - x_j) / (x_i - x_j) ) for j = 0 to m-1, j != i
    // L_i(0) = product( (-x_j) / (x_i - x_j) ) for j = 0 to m-1, j != i
    for (int i = 0; i < m; ++i) {  // 迭代選擇的 m 個份額
        ll x = v[i].first;         // 當前份額的 x 座標
        ll y = v[i].second;        // 當前份額的 y 座標

        ll num = 1;    // 拉格朗日基底多項式 L_i(0) 的分子部分
        ll denom = 1;  // 拉格朗日基底多項式 L_i(0) 的分母部分

        // 計算 L_i(0)
        for (int j = 0; j < m; ++j) {  // 迭代其他 m-1 個份額
            if (i ^ j) {               // 如果 i 不等於 j
                // 計算分子: product (-x_j)
                num = num * (mod - v[j].first) % mod;  // (mod - x_j) 等價於 -x_j 模 mod
                // 計算分母: product (x_i - x_j)
                denom = denom * (x - v[j].first + mod) % mod;  // (x_i - x_j + mod) % mod 避免負數
            }
        }

        // 計算 L_i(0) = num / denom = num * inv(denom) % mod
        ll term = num * inv(denom) % mod;

        // 將 y_i * L_i(0) 加到 secret 中
        secret = (secret + y * term % mod) % mod;
    }
    return secret % mod;  // 回傳還原的秘密值
}

int main (void) {
    cin.tie(NULL), ios_base::sync_with_stdio(false);
    ll secret = 31415926;
    cout << "Secret: " << secret << "\n\n";

    auto v = encode(secret);
    cout << "Encoded:\n";
    int k = 0;
    for (auto [x, y]: v) cout << ++k << ": (" << x << ", " << y << ")\n";

    cout << "\n";
    ll decoded = decode(v);
    cout << "Decoded: " << decoded << "\n";
    cout << "Decoded == Secret: " << (decoded == secret ? "True" : "False") << "\n";
    return 0;
}