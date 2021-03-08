//
// Created by lvhb on 2021/3/7.
//

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <queue>
#include <unordered_set>

using namespace std;

class WordTerm {
private:
    string word;
    float importance;
public:
    WordTerm();

    WordTerm(string word, float importance);

    string get_word() const;

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

string WordTerm::get_word() const {
    return this->word;
}

float WordTerm::get_importance() const {
    return this->importance;
}

bool WordTerm::operator<(const WordTerm &other) const {
    if (this->importance < other.get_importance()) {
        return true;
    } else {
        return false;
    }
}

bool WordTerm::operator>(const WordTerm &other) const {
    if (this->importance >= other.get_importance()) {
        return true;
    } else {
        return false;
    }
}

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
    if (K <= 0) {
        return vector<T>();
    }
    priority_queue<T, vector<T>, greater<>> minQ;
    for (const auto &x:vec) {
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
    unordered_map<string, float> word_scores;
    /*word_ids保存了每个单词的编号*/
    unordered_map<string, int> word_ids;
    /*keyword_num保存了每次查询的关键词数目*/
    int keyword_num;

    void calWordScores();

    unordered_map<string, unordered_set<string>> GetWordNeighbors();

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

    vector<WordTerm> GetKeywords(int p_keyword_num);

    void TransformKeywords(const vector<WordTerm> &term_vec);
};

TextRank::TextRank() {
    this->corpus.clear();
    this->keywords.clear();
    this->word_scores.clear();
    this->word_ids.clear();
    this->keyword_num = 0;
}

TextRank::TextRank(string &corpus) {
    this->corpus.clear();
    this->keywords.clear();
    this->word_scores.clear();
    this->word_ids.clear();
    this->keyword_num = 0;
    int cnt = 0;
    vector<string> str_vec = split_str(corpus, ';');
    for (auto &str:str_vec) {
        const vector<string> tem_vec = split_str(str, ' ');
        if (!tem_vec.empty()) {
            this->corpus.push_back(tem_vec);
            for (const auto &word:tem_vec) {
                if (this->word_ids.find(word) == this->word_ids.end()) {
                    this->word_ids[word] = cnt;
                    cnt++;
                }
            }
        }
    }
}

float Sigmod(float x) {
    return 1.0f / (1.0f + exp(-x));
}

void TextRank::calWordScores() {
    unordered_map<string, unordered_set<string>> word_neighbors = GetWordNeighbors();
    /*依据TF来设置word_scores的初值*/
    this->word_scores.clear();
    for (const auto &it:word_neighbors)
        this->word_scores[it.first] = Sigmod(it.second.size());

    for (int i = 0; i < MAX_ITER; i++) {
        unordered_map<string, float> new_scores_map;
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
                float neighbor_score = 0;
                if (this->word_scores.find(neighbor_word) != this->word_scores.end())
                    neighbor_score = this->word_scores[neighbor_word];
                new_score += DAMP_FACTOR * neighbor_score / (float) out_size;
            }
            if (new_scores_map.find(cur_word) == new_scores_map.end())
                new_scores_map[cur_word] = new_score;
            else
                new_scores_map[cur_word] += new_score;

            float cur_score = 0;
            if (this->word_scores.find(cur_word) != this->word_scores.end())
                cur_score = this->word_scores[cur_word];
            max_diff = max(max_diff, abs(new_score - cur_score));
        }

        this->word_scores = new_scores_map;
        if (max_diff <= MIN_DIFF)
            break;
    }
}

unordered_map<string, unordered_set<string>> TextRank::GetWordNeighbors() {
    unordered_map<string, unordered_set<string>> word_neighbors;
    //遍历每一个句子
    for (auto word_vec:this->corpus) {
        int size = word_vec.size();
        //遍历每一个单词
        for (int i = 0; i < size; i++) {
            auto cur_word = word_vec[i];
            if (word_neighbors.find(cur_word) == word_neighbors.end())
                word_neighbors[cur_word] = unordered_set<string>();

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
    for (const auto &it:this->word_scores) {
        all_terms.emplace_back(it.first, it.second);
    }
    res = topK(all_terms, MAX_KEYWORD_NUM);
    return res;
}

vector<WordTerm> TextRank::GetKeywords(int p_keyword_num) {
    if (this->keywords.empty()) {
        this->calWordScores();
        this->keywords = this->GenerateTopKeywords();
    }
    //此处使用min函数会报错
//    p_keyword_num = min(p_keyword_num, MAX_KEYWORD_NUM);
    if (p_keyword_num > MAX_KEYWORD_NUM)
        p_keyword_num = MAX_KEYWORD_NUM;
    if (p_keyword_num > this->keywords.size())
        p_keyword_num = this->keywords.size();
    this->keyword_num = p_keyword_num;
    vector<WordTerm> res;
    res.reserve(p_keyword_num);
    for (int i = 0; i < p_keyword_num; i++) {
        res.push_back(this->keywords[i]);
    }
    return res;
}

int result[TextRank::MAX_KEYWORD_NUM + 1];

void TextRank::TransformKeywords(const vector<WordTerm> &term_vec) {
    result[0] = term_vec.size();
    for (int i = 0; i < term_vec.size(); i++) {
        string word = term_vec[i].get_word();
        result[i + 1] = this->word_ids[word];
    }
    for (int i = 0; i < term_vec.size(); i++) {
        int importance = int(term_vec[i].get_importance() * 100);
        result[this->keyword_num + i + 1] = importance;
    }
}

//g++ -o text_rank.so -shared -fPIC --std=c++14 text_rank.cpp

extern "C" {
int *text_rank_wrapper(char *corpus, int keyword_num) {
    string param_corpus = corpus;
    TextRank text_rank = TextRank(param_corpus);
    vector<WordTerm> res_vec = text_rank.GetKeywords(keyword_num);
    text_rank.TransformKeywords(res_vec);
    return result;
}
}
