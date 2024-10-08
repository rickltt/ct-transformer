#include "disflu_detector.h"

namespace disfludetector
{

  bool isEnglish(const std::string& str) {
      for (char ch : str) {
          if (!std::isalpha(static_cast<unsigned char>(ch))) {
              return false;
          }
      }
      return true;
  }

  template <class ForwardIterator>
  inline size_t argmax(ForwardIterator first, ForwardIterator last)
  {
    return std::distance(first, std::max_element(first, last));
  }

  DisfluDetectorOnnx::~DisfluDetectorOnnx()
  {

    if (m_session)
    {
      delete m_session;
      m_session = nullptr;
    }
  }

  DisfluDetectorOnnx::DisfluDetectorOnnx(const char *model_path, int nNumThread)
  {
    
    std::string vocab_path = std::string(model_path) + "/vocab.json";
    LOG(INFO) << vocab_path;
    tokenizer.load_vocab(vocab_path);
    LOG(INFO) << "LoadVocab Finished";

    LOG(INFO) << model_path;
    LoadModel(model_path, nNumThread);
    LOG(INFO) << "LoadModel Finished";

  }

  void DisfluDetectorOnnx::LoadModel(const std::string &model_path, int nNumThread)
  {
    sessionOptions.SetIntraOpNumThreads(nNumThread);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.DisableMemPattern();
    sessionOptions.DisableCpuMemArena();

    std::string strModelPath = model_path + "/" + "model_quant.onnx";
    m_session = new Ort::Session(env, strModelPath.c_str(), sessionOptions);
  }


  std::string DisfluDetectorOnnx::inference(std::string input_text){

    std::vector<std::string> tokens = tokenizer.tokenize(input_text);

    int nDiff = 0;
    std::string final_result;
    for (size_t i = 0; i < tokens.size(); i += MAX_LENGTH)
    {
      nDiff = (i + MAX_LENGTH) < tokens.size() ? (0) : (i + MAX_LENGTH - tokens.size());
      std::vector<std::string> input_tokens(tokens.begin() + i, tokens.begin() + i + MAX_LENGTH - nDiff);
      final_result = final_result + inferenceMin(input_tokens);
    }
    LOG(INFO) << final_result;
    return final_result;
  }
  std::string DisfluDetectorOnnx::inferenceMin(std::vector<std::string> tokens)
  {
    tokens.insert(tokens.begin(), "<s>");
    tokens.push_back("</s>");

    std::vector<int64_t> input_ids = tokenizer.convert_tokens_to_ids(tokens);

    // for (auto &x : input_ids)
    // {
    //   std::cout << x << " ";
    // }
    // std::cout << "\n";

    std::vector<int64_t> attention_mask(input_ids.size(), 1);

    std::array<int64_t, 2> input_ids_shape{1, (int64_t)input_ids.size()};
    Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, input_ids.data(), input_ids.size(),
                                                                    input_ids_shape.data(), input_ids_shape.size());

    std::array<int64_t, 2> attention_mask_shape{1, (int64_t)attention_mask.size()};
    Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(memoryInfo, attention_mask.data(), attention_mask.size(),
                                                                         attention_mask_shape.data(), attention_mask_shape.size());

    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(input_ids_tensor));
    input_onnx.emplace_back(std::move(attention_mask_tensor));

    auto outputTensor = m_session->Run(Ort::RunOptions(), input_names.data(), input_onnx.data(), input_names.size(), output_names.data(), output_names.size());
    std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

    int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
    float *logits = outputTensor[0].GetTensorMutableData<float>();

    // std::cout << "outputCount:" << outputCount << std::endl;
    // for (const auto &elem : outputShape)
    // {
    //   std::cout << elem << " " << std::endl;
    // }
    std::vector<int> predictions;

    int num_labels = 2 * NUM_LABEL + 1;
    for (size_t i = 0; i < outputCount; i += num_labels)
    {
      int index = argmax(logits + i, logits + i + num_labels - 1);
      predictions.push_back(index);
    }
    for (const auto &pred : predictions)
    {
      std::cout << pred << " ";
    }
    std::cout << "\n";

    tokens.erase(tokens.begin());
    tokens.pop_back();

    input_ids.erase(input_ids.begin());
    input_ids.pop_back();

    predictions.erase(predictions.begin()); 
    predictions.pop_back();

    assert(input_ids.size() == predictions.size()); 
    assert(input_ids.size() == tokens.size()); 
    // 确保两个列表的长度相同
    // if (input_ids.size() != predictions.size())
    // {
    //   std::cerr << "Lists are of different lengths." << std::endl;
    // }

    // std::vector<int64_t> result;
    std::string result;
    std::string previous_token;
    for (size_t i = 0; i<predictions.size(); i++){
      if (predictions[i] == 0 || predictions[i] == 5 || predictions[i]==6){
        // result.push_back(input_ids[i]);
        std::cout << tokens[i] << " ";
        if (isEnglish(previous_token) && isEnglish(tokens[i])){
          result += " " + tokens[i];
        }
        else{
          result += tokens[i];
        }
        previous_token = tokens[i];
      }
    }
    // std::vector<std::string> decode_tokens = tokenizer.convert_ids_to_tokens(result);

    // std::string final_result;
    // std::string previous_token;
    // for(const auto &token : decode_tokens)
    // {
    //   std::cout << token << " ";
    //   if (isEnglish(previous_token) && isEnglish(token)){
    //     final_result += " " + token;
    //   }
    //   else{
    //     final_result += token;
    //   }
    //   previous_token = token;
      
    // }
    std::cout << "\n";

    return result;
  }
}