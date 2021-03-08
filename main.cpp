//
// Created by lvhb on 2021/3/7.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <queue>
#include <set>

using namespace std;

class WordTerm {
private:
    string word;
    float importance;
public:
    WordTerm();

    WordTerm(string word, float importance);

    string get_word();

    float get_importance() const;

    bool operator<(const WordTerm &other) const;

    bool operator>(const WordTerm &other) const;
};

WordTerm::WordTerm() {
    this->word = "";
    this->importance = 0;
}

WordTerm::WordTerm(string word, float importance) {
    this->word = std::move(word);
    this->importance = importance;
}

string WordTerm::get_word() {
    return this->word;
}

float WordTerm::get_importance() const {
    return this->importance;
}

bool WordTerm::operator<(const WordTerm &other) const {
    if (this->importance < other.get_importance()) {
        return true;
    }
    else {
        return false;
    }
}

bool WordTerm::operator>(const WordTerm &other) const {
    if (this->importance >= other.get_importance()) {
        return true;
    }
    else {
        return false;
    }
}

template<class K, class V>
class MapWrapper {
private:
    map<K, V> inner_map;

public:
    MapWrapper() {
        this->inner_map.clear();
    }

    explicit MapWrapper(map<K, V> param_map) {
        this->inner_map.clear();
        for (auto it:param_map) {
            this->inner_map.insert(it);
        }
    }

    bool is_exist(K key) {
        return this->inner_map.find(key) != this->inner_map.end();
    };

    void add_value(K key, V value) {
        if (this->is_exist(key)) {
            this->inner_map[key] += value;
        } else {
            this->inner_map[key] = value;
        }
    };

    void print() {
        for (const auto &it:this->inner_map) {
            cout << it.first << " " << it.second << endl;
        }
    };

    void clear() {
        this->inner_map.clear();
    }

    V get_val(K key, V default_val) {
        if (is_exist(key)) {
            return this->inner_map[key];
        } else {
            return default_val;
        }
    }

    vector<K> get_all_keys() {
        vector<K> res;
        for (auto& it:this->inner_map) {
            res.push_back(it.first);
        }
        return res;
    }
};

vector<string> split_str(string &ori_str, char separator) {
    vector<int> pos_vec;
    ori_str = separator + ori_str + separator;
    int len = ori_str.length();
    for (int i = 0; i < len; i++)
        if (ori_str[i] == separator) {
            pos_vec.push_back(i);
        }

    vector<string> res;
    int size = pos_vec.size();
    for (int i = 0; i < size - 1; i++) {
        string sub = ori_str.substr(pos_vec[i] + 1, pos_vec[i + 1] - pos_vec[i] - 1);
        if (!sub.empty())
            res.push_back(sub);
    }

    return res;
}

/*
 * 从vec中寻找最大的K个数
 * */
template<class T>
vector<T> topK(const vector<T> &vec, int K) {
    if (K <= 0){
        return vector<T>();
    }
    priority_queue<T, vector<T>, greater<>> minQ;
    for (auto x:vec) {
        if (minQ.size() < K) {
            minQ.push(x);
        }
            //minQ.top()是minQ中最小的数
        else if (minQ.top() < x) {
            minQ.pop();
            minQ.push(x);
        }
    }

    vector<T> res;
    while (!minQ.empty()) {
        res.push_back(minQ.top());
        minQ.pop();
    }
    sort(res.begin(), res.end());
    reverse(res.begin(), res.end());
    return res;
}

class TextRank {
private:
    /*corpus是迭代训练的语料*/
    vector<vector<string>> corpus;
    /*keywords是训练得到的关键词*/
    vector<WordTerm> keywords;
    /*word_weights保存了每个单词的分数，分数越大越是关键词*/
    MapWrapper<string, float> word_scores;

    void calWordScores();

    map<string, set<string>> GetWordNeighbors();

    vector<WordTerm> GenerateTopKeywords();

public:
    //阻尼系数，一般取值为0.85
    static constexpr float DAMP_FACTOR = 0.85;
    //最大迭代次数
    static const int MAX_ITER = 200;
    //两次迭代之间权重收敛的阈值
    static constexpr float MIN_DIFF = 0.001;
    //TextRank模型的窗口大小
    static const int WINDOW_SIZE = 4;
    //关键词最大的数量
    static const int MAX_KEYWORD_NUM = 30;

    TextRank();

    explicit TextRank(string &corpus);

    vector<WordTerm> GetKeywords(int keyword_num);

    static string TransformKeywords2Str(const vector<WordTerm>& term_vec);
};

TextRank::TextRank() {
    this->corpus.clear();
    this->keywords.clear();
    this->word_scores.clear();
}

TextRank::TextRank(string &corpus) {
    this->corpus.clear();
    this->keywords.clear();
    this->word_scores.clear();
    vector<string> str_vec = split_str(corpus, ';');
    for (auto &str:str_vec) {
        const vector<string> tem_vec = split_str(str, ' ');
        if (!tem_vec.empty()) {
            this->corpus.push_back(tem_vec);
        }
    }
}

float sigmod(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void TextRank::calWordScores() {
    map<string, set<string>> word_neighbors = GetWordNeighbors();
    /*依据TF来设置word_scores的初值*/
    this->word_scores.clear();
    for (const auto &it:word_neighbors) {
        this->word_scores.add_value(it.first, sigmod(it.second.size()));
    }

    for (int i = 0; i < MAX_ITER; i++) {
        MapWrapper<string, float> new_scores;
        float max_diff = 0;
        //遍历每一个单词cur_word
        for (const auto &it:word_neighbors) {
            string cur_word = it.first;
            auto all_neighbors = it.second;
            float new_score = 1 - DAMP_FACTOR;
            //遍历cur_word的每一个邻居
            for (auto &neighbor_word:all_neighbors) {
                int out_size = word_neighbors[neighbor_word].size();
                if (cur_word == neighbor_word || out_size == 0)
                    continue;
                float neighbor_score = this->word_scores.get_val(neighbor_word, 0);
                new_score += DAMP_FACTOR * neighbor_score / (float) out_size;
            }
            new_scores.add_value(cur_word, new_score);

            float cur_score = this->word_scores.get_val(cur_word, 0);
            max_diff = max(max_diff, abs(new_score - cur_score));
        }

        this->word_scores = new_scores;
        if (max_diff <= MIN_DIFF)
            break;
    }
}

map<string, set<string>> TextRank::GetWordNeighbors() {
    map<string, set<string>> word_neighbors;
    //遍历每一个句子
    for (auto word_vec:this->corpus) {
        int size = word_vec.size();
        //遍历每一个单词
        for (int i = 0; i < size; i++) {
            auto cur_word = word_vec[i];
            if (word_neighbors.find(cur_word) == word_neighbors.end())
                word_neighbors[cur_word] = set<string>();

            for (int j = max(0, i - WINDOW_SIZE); j <= i + WINDOW_SIZE && j < size; j++) {
                if (cur_word != word_vec[j]) {
                    word_neighbors[cur_word].insert(word_vec[j]);
                }
            }
        }
    }
    return word_neighbors;
}

vector<WordTerm> TextRank::GenerateTopKeywords() {
    vector<WordTerm> res;
    vector<WordTerm> all_terms;
    for (const auto &key:this->word_scores.get_all_keys()) {
        auto val = this->word_scores.get_val(key, 0);
        all_terms.emplace_back(key, val);
    }
    res = topK(all_terms, MAX_KEYWORD_NUM);
    return res;
}

vector<WordTerm> TextRank::GetKeywords(int keyword_num) {
    if (this->keywords.empty()) {
        this->calWordScores();
        this->keywords = this->GenerateTopKeywords();
    }
    //此处使用min函数会报错
//    keyword_num = min(keyword_num, MAX_KEYWORD_NUM);
    if (keyword_num > MAX_KEYWORD_NUM)
        keyword_num = MAX_KEYWORD_NUM;
    vector<WordTerm> res;
    res.reserve(keyword_num);
    for (int i = 0; i < keyword_num; i++) {
        res.push_back(this->keywords[i]);
    }
    return res;
}

string TextRank::TransformKeywords2Str(const vector<WordTerm> &term_vec) {
    string res_str;
    for (auto term:term_vec) {
        res_str += term.get_word() + " ";
    }
    res_str += ";";
    for (const auto &term:term_vec) {
        res_str += to_string(term.get_importance()) + " ";
    }
    return res_str;

}

string get_test(){
    string s = "";
    for(int i=0;i<10;i++){
        s+=to_string(i);
    }
    return s;
}


//g++ -o text_rank.so -shared -fPIC --std=c++14 text_rank.cpp
string res_str;

extern "C" {
char *text_rank_wrapper(char *corpus, int keyword_num) {
    string param_corpus = corpus;
    TextRank text_rank = TextRank(param_corpus);
    vector<WordTerm> res_vec = text_rank.GetKeywords(keyword_num);
    res_str = text_rank.TransformKeywords2Str(res_vec);
    char *t = (char *) res_str.c_str();
    return t;
}

}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
