#pragma once

#include <iostream>
#include <numeric>
#include <fstream> 
#include <map> 
#include <vector>
#include <cmath>
#include <random>
#include <unordered_set>
#include <algorithm>
#include <codecvt>
#include <assert.h>
#include <glog/logging.h>
#include "nlohmann/json.hpp"

namespace disflu_tokenizer
{
    using json = nlohmann::json;
    class Tokenizer
    {
    public:
        Tokenizer();
        ~Tokenizer();
        std::vector<std::string> convert_ids_to_tokens(const std::vector<int64_t> &token_ids);
        std::vector<int64_t> convert_tokens_to_ids(const std::vector<std::string> &tokens);
        std::vector<std::string> tokenize(const std::string &text);
        void load_vocab(std::string vocab_path);
    private:
        //vocab
        std::vector<std::string> vocab;
   
        json id_to_token;
        json token_to_id;
        std::string sentence_start_token;
        std::string sentence_end_token;
        std::string padding_token;
        std::string unk_token;

    };
}
