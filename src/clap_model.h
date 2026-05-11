
#ifndef CLAP_TILDE_CLAP_MODEL_H
#define CLAP_TILDE_CLAP_MODEL_H

// Wraps the TorchScript model exported by scripts/export_clap.py.
//
// TorchScript interface:
//   get_sr()                                                     -> int
//   get_seglen()                                                 -> int
//   get_max_text_length()                                        -> int
//   encode_text_from_tokens(input_ids [N,L], attn_mask [N,L])   -> Tensor [N, 512]
//   forward(audio [1,1,N], text_embs [M,512])                   -> Tensor [M]

#include <torch/script.h>
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <chrono>

#include "bpe_tokenizer.h"


struct ClassificationResult {
    std::vector<float>       distribution;
    double                   inference_latency_ms;
    std::vector<std::string> class_names;   // filled by ClapClassifier before enqueue
};


struct IClapModel {
    virtual ~IClapModel() = default;
    virtual at::Tensor encode_text(const std::vector<std::string>& class_names) = 0;
    // Returns [1, 512] normalised audio embedding, or empty tensor if unsupported.
    virtual at::Tensor encode_audio(std::vector<float> audio) = 0;
    virtual ClassificationResult classify(std::vector<float> audio,
                                          const at::Tensor& text_embs) = 0;
    virtual int get_sample_rate()    const = 0;
    virtual int get_segment_length() const = 0;
};


class ClapModel : public IClapModel {
public:
    static const inline std::string FORWARD_METHOD            = "forward";
    static const inline std::string ENCODE_TEXT_METHOD        = "encode_text_from_tokens";
    static const inline std::string SR_METHOD                 = "get_sr";
    static const inline std::string SEGLEN_METHOD             = "get_seglen";
    static const inline std::string MAX_TEXT_LENGTH_METHOD    = "get_max_text_length";

    // model_path:      path to clap_tilde.ts
    // tokenizer_dir:   directory containing vocab.json + merges.txt
    explicit ClapModel(const std::string& model_path,
                       const std::string& tokenizer_dir,
                       torch::DeviceType device)
        : m_device(device)
    {
        at::init_num_threads();

        m_model = torch::jit::load(model_path);
        m_model.eval();
        m_model.to(m_device);

        m_sample_rate    = call_int_method(SR_METHOD);
        m_segment_length = call_int_method(SEGLEN_METHOD);
        int max_text_len = call_int_method(MAX_TEXT_LENGTH_METHOD);

        m_tokenizer = std::make_unique<BPETokenizer>(tokenizer_dir, max_text_len);
    }

    // Returns [1, 512] audio embedding, or empty tensor for old .ts without encode_audio.
    at::Tensor encode_audio(std::vector<float> audio_samples) override {
        try {
            auto audio_tensor = torch::from_blob(
                audio_samples.data(),
                {1, 1, static_cast<long long>(audio_samples.size())},
                torch::kFloat32).clone().to(m_device);
            auto result = m_model.get_method("encode_audio")({audio_tensor}).toTensor();
            return result.to(torch::kCPU);
        } catch (...) {
            return {};
        }
    }

    // Encode class name strings → text embedding tensor [N, 512].
    // Tokenisation (BPE) is done in C++; token tensors are passed to TorchScript.
    at::Tensor encode_text(const std::vector<std::string>& class_names) override {
        auto [input_ids, attention_mask] = m_tokenizer->encode(class_names);
        input_ids      = input_ids.to(m_device);
        attention_mask = attention_mask.to(m_device);

        std::vector<torch::jit::IValue> inputs = {input_ids, attention_mask};
        auto result = m_model.get_method(ENCODE_TEXT_METHOD)(inputs).toTensor();
        return result.to(torch::kCPU);
    }

    // Run the audio forward pass with pre-computed text embeddings.
    // audio_samples: float32, exactly m_segment_length samples at 48 kHz
    ClassificationResult classify(std::vector<float> audio_samples,
                                  const at::Tensor&  text_embeddings) override {
        auto audio_tensor = torch::from_blob(
            audio_samples.data(),
            {1, 1, static_cast<long long>(audio_samples.size())},
            torch::kFloat32).clone();

        audio_tensor = audio_tensor.to(m_device);
        auto text_on_device = text_embeddings.to(m_device);

        std::vector<torch::jit::IValue> inputs = {audio_tensor, text_on_device};

        auto t1 = std::chrono::high_resolution_clock::now();
        auto out = m_model.get_method(FORWARD_METHOD)(inputs).toTensor();
        auto t2 = std::chrono::high_resolution_clock::now();

        out = out.to(torch::kCPU);

        auto latency_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
        return ClassificationResult{tensor_to_vector(out),
                                    static_cast<double>(latency_ns) / 1e6};
    }

    int get_sample_rate()    const override { return m_sample_rate; }
    int get_segment_length() const override { return m_segment_length; }

private:
    int call_int_method(const std::string& method) {
        return m_model.get_method(method)(std::vector<c10::IValue>()).to<int>();
    }

    static std::vector<float> tensor_to_vector(const at::Tensor& t) {
        std::vector<float> v;
        v.reserve(static_cast<std::size_t>(t.numel()));
        auto ptr = t.contiguous().data_ptr<float>();
        std::copy(ptr, ptr + t.numel(), std::back_inserter(v));
        return v;
    }

    torch::DeviceType     m_device;
    torch::jit::Module    m_model;
    int                   m_sample_rate;
    int                   m_segment_length;
    std::unique_ptr<BPETokenizer> m_tokenizer;
};


#endif //CLAP_TILDE_CLAP_MODEL_H
