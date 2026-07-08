#pragma once
#include "Model.h"
#include <fstream>
#include "STrack.h"
#include <atomic>
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <mutex>

// #define TIME_DEBUG
// #define ALL_TIMEDEBUG

namespace TrackerBySAM2 {


enum TRACKTYPEBYSAM2 {
    SingleTrack = 0,
    MultiTrack,
};

enum TensorRtType {
    ImageEncoder = 1,    // 显式指定值
    MemoryAttention,      // 默认从上一个值加1，即2
    ImageDecoderFirst,        // 默认从上一个值加1，即3
    ImageDecoderSecond,
    MemoryEncoder,
    ImageDecoder
};

bool found_tlwh(const float *ptr,std::vector<float>& tlwh_);
void print_tensor_shape(Ort::Value& tensor_value);

inline std::string get_timestamp_filename() {
    // 获取当前时间（精确到毫秒）
    auto now = std::chrono::system_clock::now();
    auto now_ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
    auto epoch = now_ms.time_since_epoch();
    auto value = std::chrono::duration_cast<std::chrono::milliseconds>(epoch);
    long long milliseconds = value.count();

    // 转换为本地时间（年月日_时分秒_毫秒）
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm = *std::localtime(&now_c);
    
    // 格式化文件名
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y%m%d_%H%M%S_") 
        << std::setfill('0') << std::setw(3) << (milliseconds % 1000)
        << "_start.jpg";
    
    return oss.str();
}


// tensorrt代码
// 自定义日志记录器类
class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 只打印警告和错误信息
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

class TensorRTInference {
public:
    TensorRTInference(const std::string& engineFilePath, int gpuId);
    ~TensorRTInference() = default;

    //infer is for mem_attention infer
    bool MemAttentioninfer(std::vector<Ort::Value>& input_tensor, float* ptr_tensorrt_mem_attention_data);

    void infer_ImageEncoder(void* gpuPtr_image, std::vector<int>& batch_size_info, std::vector<std::vector<float>>& outputHostData, TRACKTYPEBYSAM2 trackType); 
    void infer_ImageDecoderFirst(std::vector<float>& point_val,
                                 std::vector<int>&   point_labels,
                                 std::vector<Ort::Value>& mem_attention_out,
                                 std::vector<std::vector<float>>& outputDataForImageDecoder_First);

    void infer_ImageDecoderSecond(sKfOut& result_kf, std::vector<std::vector<float>>& outputDataForImageDecoder_Second);
    void infer_MemoryDecoder(std::vector<Ort::Value>& MemoryDecoderIn, std::vector<std::vector<float>>& outputDataForMmeEncoder, std::vector<void *>&mem_encoder_inStr);


    void infer_ImageDecoder(std::vector<float>& point_val,
        std::vector<int>&   point_labels,
        std::vector<Ort::Value>& mem_attention_out,
        std::vector<std::vector<float>>& outputDataForImageDecoder_Second,
        std::vector<void *>&inputStr);
    
    void allocateInputAndOutputMemory();
    void releaseInputAndOutputMemory();

    int batch_size = 0;
    int obj_size = 16;               //mem_attention size
    int mem_feature_size = 7;        //mem_attention size
    TensorRtType TRTtype;


    std::vector<void*> outputData;
    std::vector<void*> inputData;
    std::vector<size_t> inputSizes;
    std::vector<size_t> outputSizes;
    std::vector<void*> device_buffers;

private:
    std::vector<char> loadEngine(const std::string& engineFilePath);
    void* allocateDeviceMemory(size_t size);
    void freeDeviceMemory(void* deviceMemory);

    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    cudaStream_t stream;

    int gpuId = 0;
};

struct ParamsSam2{
    uint type = 0; // 0使用box，1使用point
    cv::Rect prompt_box;
    cv::Point prompt_point;
};


class Sam2Singleton : public yo::Model 
{
    private:
        static std::unique_ptr<Sam2Singleton> instance;      // 唯一实例
        static std::mutex mtx;                      // 互斥锁，用于线程安全的初始化
        static std::atomic<bool> is_initialized;    // 使用原子变量来确保只初始化一次
    public:
        // 获取单例实例
        static Sam2Singleton& getInstance() {
            if (!instance) {
                std::lock_guard<std::mutex> lock(mtx);  // 线程安全的创建唯一实例
                if (!instance) {  // 双重检查锁定
                    instance.reset(new Sam2Singleton());
                }
            }
            return *instance;
        }
    
        // 删除拷贝构造函数和赋值运算符，确保不能拷贝单例
        Sam2Singleton(const Sam2Singleton&) = delete;
        Sam2Singleton& operator=(const Sam2Singleton&) = delete;
    
        // 其它成员函数和变量保持不变
        int setparms(std::vector<ParamsSam2>& parms);
        bool initialize(std::vector<std::string>& tensorrt_paths, int trackType, int gpuId) override;
        bool inference(cv::Mat &image) override;
    
        void sam2Process(std::vector<cv::Mat> &images, cv::Rect& rectInfoIn, std::vector<cv::Rect>& rectOut);
        int setparms_InSam2Process(std::vector<ParamsSam2>& parms);
        int setparms_AllocateBatch_One();

        // 必要的成员变量
        int sizeOfMemfeature = 0;
        int sizeOfObj = 0;
        int combineMemAndObj_size = 0;
        cv::Rect LastRect;
    
        TensorRTInference *ImageEndocer_engine;
        TensorRTInference *mem_attention_engine_mem7_obj16;
        TensorRTInference *ImageDecoderFirst_engine;
        TensorRTInference *ImageDecoderEnd_engine;
        TensorRTInference *MemoryEncoder_engine;
        TensorRTInference *ImageDecoder_engine;

    
        std::unique_lock<std::mutex> getLock() {
            return std::unique_lock<std::mutex>(m_mutex);  // 返回一个锁
        }

        ~Sam2Singleton() {
            // 清理资源代码
        }
    protected:
        Sam2Singleton() { 
            // 初始化代码
        }
    
    private:
        static bool m_isInitialized;          // 初始化状态
        static std::once_flag m_onceFlag;    // std::call_once 用来确保初始化只执行一次

        static const size_t BUFFER_SIZE = 15;

        static std::mutex m_mutex;

        struct SubStatus {
            std::vector<Ort::Value> maskmem_features;
            std::vector<Ort::Value> maskmem_pos_enc;
            std::vector<Ort::Value> temporal_code;
            std::vector<float> iou;
            float obj_score;
            std::unique_ptr<std::vector<float>> data_holder_maskmem_features;
            std::unique_ptr<std::vector<float>> data_holder_maskmem_pos_enc;
            std::unique_ptr<std::vector<float>> data_holder_temporal_code;
        };
    
        struct Obj_Ptr_Status {
            std::vector<Ort::Value> obj_ptr;
            std::vector<float> iou;
            float obj_score;
            std::unique_ptr<std::vector<float>> data_holder;
        };
    
        struct InferenceStatus {
            int32_t current_frame = 0;
            std::vector<Obj_Ptr_Status> obj_ptr_first;
            std::vector<SubStatus> status_first;
            std::vector<yo::FixedSizeQueue<SubStatus, 7 - 2>> status_recent;
            std::vector<yo::FixedSizeQueue<Obj_Ptr_Status, BUFFER_SIZE - 1>> obj_ptr_recent;
            yo::FixedSizeQueue<SubStatus, 1> last_memoryFeature;
            yo::FixedSizeQueue<Obj_Ptr_Status, 1> last_objPtr;
            int stable_frames = 0;
            int stable_frames_threshold = 15;
            float stable_ious_threshold = 0.3;
            float kf_score_weight = 0.15;
            float memory_bank_iou_threshold = 0.5;
            float memory_bank_obj_score_threshold = 0;
        };
    
        bool is_inited = false;
        cv::Mat* ori_img = nullptr;
        std::vector<cv::Mat> input_images;
        std::vector<ParamsSam2> parms;
        int batch_size;
        InferenceStatus infer_status;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<STrackInSam2> tracked_stracks;
    
        // Protected methods for processing
        void preprocess(cv::Mat &image) override;
        void postprocess(std::vector<Ort::Value>& output_tensors) override;
    
        std::vector<Ort::Value>img_encoder_infer(std::vector<Ort::Value>&);
        std::vector<Ort::Value>img_decoder_infer(std::vector<Ort::Value>&);
        std::vector<Ort::Value> mem_attention_infer(std::vector<Ort::Value>&);
        std::vector<Ort::Value> img_decoder_inferStart(std::vector<Ort::Value>&);
        std::vector<Ort::Value> img_decoder_inferEnd(sKfOut&);
        void mem_attention_infer(std::vector<Ort::Value>&img_encoder_out, std::vector<Ort::Value>& mem_attention_out);

        void img_decoder_KF(std::vector<Ort::Value>& decoderKfIn, sKfOut& kftmp);
        void creatImageDecoder_First(std::vector<Ort::Value> &ImageDecoder_First_out);
        void creatPointInput(std::vector<float> &point_val, std::vector<int> &point_labels);
        void creatTensorImageDecoder_Second(std::vector<Ort::Value> &img_decoder_out);
        void creatTensorMemEncoder(std::vector<Ort::Value> &MemEncoder_out);
        void createTensorImgEncoder(std::vector<Ort::Value>& img_encoder_out);
        void gpu_preprocess_sync(
            const cv::Mat& input,
            float* output_ptr  // 预分配内存地址：image_preprocess_data.data()
        );
        void gpu_preprocess_async(
            const cv::Mat& input,
            float* output_ptr  // 预分配内存地址：image_preprocess_data.data()
        );
        std::vector<float> get_best_iou();
        cv::Rect getOutBBoxFromSam2Out(cv::Mat& outimg);

        //提取初始化 防止局部创建根据数据指针创建tensor的时候 数据指针被清理了 
        //当使用CreateTensor并传递数据指针时，需要确保该内存在张量使用期间有效，并且用户需要负责管理内存。
        std::vector<float> low_res_masks_data;           // [batch_size, 1, 256, 256]
        std::vector<float> high_res_masks_data;          // [batch_size, 1, 1024, 1024]
        std::vector<float> sam_output_data;              // [batch_size, 256]
        std::vector<float> tensorrt_mem_attention_data;
        std::vector<std::vector<float>> outputHostData;
        std::vector<std::vector<float>> outputDataForImageDecoder_First;
        std::vector<std::vector<float>> outputDataForImageDecoder_Second;
        std::vector<std::vector<float>> outputDataForMmeEncoder;
        std::vector<float> image_preprocess_data;

        std::vector<float> obj_ptrs; // first+recent // 16*256 * batch_size //todo零拷贝
        std::vector<float> maskmem_features_;     //todo零拷贝
        std::vector<float> maskmem_pos_enc_;     //todo零拷贝

        std::vector<cv::Scalar> g_vcolors = {
            cv::Scalar(255, 0, 0),   // 蓝色
            cv::Scalar(0, 255, 0),   // 绿色
            cv::Scalar(0, 0, 255),   // 红色
            cv::Scalar(255, 255, 0), // 青色
            cv::Scalar(255, 0, 255), // 品红
            cv::Scalar(0, 255, 255), // 黄色
            cv::Scalar(128, 0, 128), // 紫色
            cv::Scalar(128, 128, 0), // 橄榄色
            cv::Scalar(0, 128, 128), // 青绿色
            cv::Scalar(128, 128, 128) // 灰色
        };

        int gpuId = 0;

        /* begin for preprocess */
        cv::cuda::Stream stream;
        std::vector<cv::cuda::GpuMat> gpu_buffers;
        std::vector<cv::cuda::GpuMat> channel_buffers;
        cv::Mat page_locked_buffer;
        /* end   for preprocess */

        /* begin for postprocess */
        cv::Mat convert8UC1_mat;
        cv::Mat bin_mat;

        cv::cuda::GpuMat d_outimg;
        cv::cuda::GpuMat d_bin;

        uchar* pinned_ptr = nullptr;
        cv::Mat page_locked_bufferPostProcess;
        /* end   for postprocess */

        TRACKTYPEBYSAM2 trackType;
};

}