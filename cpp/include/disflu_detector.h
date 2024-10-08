#pragma once

#define MAX_LENGTH 40
#define NUM_LABEL 3

#include "tokenizer.h"
#include "onnxruntime_cxx_api.h" // NOLINT

namespace disfludetector{

class DisfluDetectorOnnx{

    public:
        DisfluDetectorOnnx(const char* model_path, int nNumThread);
        ~DisfluDetectorOnnx();
        std::string inference(std::string input_text);
        std::string inferenceMin(std::vector<std::string> tokens);

    private:

        disflu_tokenizer::Tokenizer tokenizer;
        Ort::Session* m_session;
        void LoadModel(const std::string& model_path, int nNumThread);
        Ort::SessionOptions sessionOptions = Ort::SessionOptions();
        Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "DisfluTest");
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // model
        std::vector<int64_t> input_ids;
        std::vector<const char*> input_names{"input_ids","attention_mask"};
        std::vector<const char*> output_names{"logits"};

};

}