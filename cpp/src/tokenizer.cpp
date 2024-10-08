
#include "tokenizer.h"

// // 将UTF-8字符串转换为宽字符
// std::wstring utf8_to_wstring(const std::string &str)
// {
//     std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
//     return converter.from_bytes(str);
// }

// // 将宽字符转换为UTF-8字符串
// std::string wstring_to_utf8(const std::wstring &wstr)
// {
//     std::wstring_convert<std::codecvt_utf8<wchar_t>> converter;
//     return converter.to_bytes(wstr);
// }


namespace disflu_tokenizer
{
    Tokenizer::Tokenizer()
    {
        // load_vocab(vocab_path);
        sentence_start_token = "<s>";
        sentence_end_token = "</s>";
        padding_token = "<pad>";
        unk_token = "<unk>";
    }

    Tokenizer::~Tokenizer()
    {
    }

    std::vector<std::string> Tokenizer::convert_ids_to_tokens(const std::vector<int64_t> &token_ids)
    {
        std::vector<std::string> tokens;
        for (int64_t token_id : token_ids)
        {
            tokens.push_back(id_to_token[token_id]);
        }
        return tokens;
    }

    std::vector<int64_t> Tokenizer::convert_tokens_to_ids(const std::vector<std::string> &tokens)
    {
        std::vector<int64_t> ids;
        for (const std::string &token : tokens)
        {
            std::string lower_token = token; // To lower case
            std::transform(lower_token.begin(), lower_token.end(), lower_token.begin(), ::tolower);
            if (token_to_id.contains(lower_token))
            {
                ids.push_back(token_to_id[lower_token]);
            }
            else
            {
                ids.push_back(token_to_id[unk_token]);
            }
        }
        return ids;
    }
    std::vector<std::string> Tokenizer::tokenize(const std::string &text)
    {
        // std::string lower_text = text; // To lower case
        // std::transform(lower_text.begin(), lower_text.end(), lower_text.begin(), ::tolower);
        // LOG(INFO) << "lower text: "<< lower_text;
        std::vector<std::string> tokenize_result;
        std::vector<std::string> result_tokens;
        std::string current_token;

        size_t i = 0;
        while (i < text.size())
        {
            unsigned char charac = text[i];
            if (charac < 0x80)
            { // ASCII 字符
                if (isalpha(charac) || isdigit(charac))
                {
                    current_token += charac; // 英文字符
                }
                else
                {
                    if (!current_token.empty())
                    {
                        tokenize_result.push_back(current_token);
                        current_token.clear();
                    }
                    if (!isspace(charac))
                    {
                        tokenize_result.push_back(std::string(1, charac)); // 其他符号
                    }
                }
                i++;
            }
            else
            { // 非 ASCII 字符
                // UTF-8 中文字符占 3 到 4 个字节
                if ((charac & 0xF0) == 0xE0)
                { // 3 字节字符
                    if (!current_token.empty())
                    {
                        tokenize_result.push_back(current_token);
                        current_token.clear();
                    }
                    tokenize_result.push_back(text.substr(i, 3)); // 取出中文字符
                    i += 3;
                }
                else if ((charac & 0xF8) == 0xF0)
                { // 4 字节字符
                    if (!current_token.empty())
                    {
                        tokenize_result.push_back(current_token);
                        current_token.clear();
                    }
                    tokenize_result.push_back(text.substr(i, 4)); // 取出中文字符
                    i += 4;
                }
                else
                {
                    // tokenize_result.push_back("<unk>");
                    // 处理无效的 UTF-8 字符（可以根据需求添加错误处理）
                    i++;
                }
            }
        }

        if (!current_token.empty())
        { // 处理最后一个 token
            tokenize_result.push_back(current_token);
        }

        // Process token result
        for (const auto& token : tokenize_result) {
            // LOG(INFO) << token;
            if (token_to_id.contains(token)) {
                result_tokens.push_back(token);
            } else {
                result_tokens.push_back(unk_token);
            }
        }
        return tokenize_result;
        // return result_tokens;
    }


    void Tokenizer::load_vocab(std::string vocab_path)
    {
        std::ifstream file(vocab_path);
        json json_vocab;
        file >> json_vocab;

        for (const auto &token : json_vocab)
        {
            vocab.push_back(token);
        }
        LOG(INFO) << "vocab size: " << vocab.size();
        for (size_t idx = 0; idx < vocab.size(); ++idx)
        {
            id_to_token[idx] = vocab[idx];
            token_to_id[vocab[idx]] = idx;
        }
    }
}
