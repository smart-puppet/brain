#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdio>
#include <stdexcept>
#include <string>

#include "ggml.h"
#include "parakeet_capi.h"
#include "ggml_graph.hpp"

namespace py = pybind11;

// parakeet.cpp links ggml with CUDA graphs enabled; streaming STT fires many
// small graphs per audio chunk and ggml logs every "CUDA graph warmup complete"
// at DEBUG to stderr. Filter to WARN+ so component tests stay readable.
static void parakeet_ggml_log_callback(ggml_log_level level, const char* text,
                                     void* user_data) {
  (void)user_data;
  if (level == GGML_LOG_LEVEL_DEBUG || level == GGML_LOG_LEVEL_INFO ||
      level == GGML_LOG_LEVEL_CONT) {
    return;
  }
  fputs(text, stderr);
  fflush(stderr);
}

class ParakeetStream {
 public:
  explicit ParakeetStream(parakeet_stream* stream) : stream_(stream) {}
  ~ParakeetStream() { close(); }

  ParakeetStream(const ParakeetStream&) = delete;
  ParakeetStream& operator=(const ParakeetStream&) = delete;

  void close() {
    if (stream_) {
      parakeet_capi_stream_free(stream_);
      stream_ = nullptr;
    }
  }

  py::tuple feed(py::array_t<float> pcm, int sample_rate) {
    (void)sample_rate;
    if (!stream_) {
      throw std::runtime_error("parakeet stream is closed");
    }
    py::buffer_info buf = pcm.request();
    if (buf.ndim != 1) {
      throw std::runtime_error("PCM must be 1-D float32 array");
    }
    int eou = 0;
    char* text = parakeet_capi_stream_feed(
        stream_, static_cast<const float*>(buf.ptr), static_cast<int>(buf.size), &eou);
    if (!text) {
      throw std::runtime_error("parakeet stream feed failed");
    }
    std::string out(text);
    parakeet_capi_free_string(text);
    return py::make_tuple(out, eou != 0);
  }

  std::string finalize() {
    if (!stream_) {
      return "";
    }
    char* text = parakeet_capi_stream_finalize(stream_);
    if (!text) {
      return "";
    }
    std::string out(text);
    parakeet_capi_free_string(text);
    return out;
  }

 private:
  parakeet_stream* stream_ = nullptr;
};

class ParakeetContext {
 public:
  explicit ParakeetContext(const std::string& model_path) {
    ctx_ = parakeet_capi_load(model_path.c_str());
    if (!ctx_) {
      throw std::runtime_error("Failed to load parakeet model: " + model_path);
    }
  }

  ~ParakeetContext() { close(); }

  ParakeetContext(const ParakeetContext&) = delete;
  ParakeetContext& operator=(const ParakeetContext&) = delete;

  void close() {
    if (ctx_) {
      parakeet_capi_free(ctx_);
      ctx_ = nullptr;
    }
  }

  void set_att_context(int left, int right) {
    if (!ctx_) {
      throw std::runtime_error("parakeet context is closed");
    }
    if (parakeet_capi_set_att_context(ctx_, left, right) != 0) {
      const char* err = parakeet_capi_last_error(ctx_);
      throw std::runtime_error(err && err[0] ? err : "set_att_context failed");
    }
  }

  ParakeetStream* stream_begin_lang(const std::string& lang) {
    if (!ctx_) {
      throw std::runtime_error("parakeet context is closed");
    }
    parakeet_stream* stream = parakeet_capi_stream_begin_lang(ctx_, lang.c_str());
    if (!stream) {
      const char* err = parakeet_capi_last_error(ctx_);
      throw std::runtime_error(err && err[0] ? err : "stream_begin failed");
    }
    return new ParakeetStream(stream);
  }

 private:
  parakeet_ctx* ctx_ = nullptr;
};

PYBIND11_MODULE(puppet_parakeet, m) {
  m.doc() = "pybind11 bindings for parakeet.cpp streaming STT";

  ggml_log_set(parakeet_ggml_log_callback, nullptr);

  py::class_<ParakeetStream>(m, "Stream")
      .def("feed", &ParakeetStream::feed)
      .def("finalize", &ParakeetStream::finalize)
      .def("close", &ParakeetStream::close);

  py::class_<ParakeetContext>(m, "Context")
      .def(py::init<const std::string&>())
      .def("set_att_context", &ParakeetContext::set_att_context)
      .def("stream_begin_lang", &ParakeetContext::stream_begin_lang, py::return_value_policy::take_ownership)
      .def("close", &ParakeetContext::close);

  m.def("load", [](const std::string& path) { return new ParakeetContext(path); },
        py::return_value_policy::take_ownership);

  m.def(
      "set_num_threads",
      [](int n) { pk::set_num_threads(n); },
      py::arg("n"),
      "Set ggml compute thread count for parakeet (0 = use per-component defaults).");
}
