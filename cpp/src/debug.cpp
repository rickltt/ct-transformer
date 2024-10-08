#include <iostream>
#include <glog/logging.h>

#include "disflu_detector.h"
int main(int argc, char *argv[])
{
    const char *model_path = "model_checkpoint";
    auto detector = disfludetector::DisfluDetectorOnnx(model_path, 1);

    std::string text1 = "因为台为为为你想每个人一台嘛1400个了，就一千四百台嘛啊，每年都一一台。";
    std::string text1_result = detector.inference(text1);
    LOG(INFO) << text1_result;
    // std::string vocab_path = "D:\\translate_test\\test_dll\\ct_transformer\\model_checkpoint\\vocab.json";
    // auto tokenizer = disflu_tokenizer::Tokenizer(vocab_path);

    // std::string text = "Hello, 你好！This 2025 40 is a test.";
    // LOG(INFO) << text;
    // std::vector<std::string> tokens = tokenizer.tokenize(text);
    // std::vector<int64_t> token_ids = tokenizer.convert_tokens_to_ids(tokens);
    // std::vector<std::string> convert_tokens = tokenizer.convert_ids_to_tokens(token_ids);
    // for (auto &token: tokens){
    //     LOG(INFO) << token;
    // }

    // for (auto &token_id: token_ids){
    //     LOG(INFO) << token_id;
    // }

    // for (auto &token: convert_tokens){
    //     LOG(INFO) << token;
    // }

    return 0;
}
