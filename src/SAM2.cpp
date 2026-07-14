#include "SAM2.h"
#include <opencv2/dnn.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>

namespace TrackerBySAM2 {

std::unique_ptr<Sam2Singleton> Sam2Singleton::instance = nullptr;  // 初始化静态成员
std::mutex Sam2Singleton::mtx;                           // 互斥锁初始化
std::atomic<bool> Sam2Singleton::is_initialized(false);   // 初始化标志为false
std::mutex Sam2Singleton::m_mutex;

// 指定初始化路径 + gpuId + 跟踪类型 单目标/多目标
bool Sam2Singleton::initialize(std::vector<std::string>& tensorrt_paths, int trackType, int gpuId){
    this->gpuId = gpuId;
    this->trackType = static_cast<TRACKTYPEBYSAM2>(trackType);

    cudaError_t cudaStatus = cudaSetDevice(gpuId);

    if (is_initialized.load()) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mtx);

    if (is_initialized.load()) {
        return true;
    }

    this->is_inited = true;
    mem_attention_engine_mem7_obj16 = new TensorRTInference(tensorrt_paths[0], gpuId);

    ImageEndocer_engine      = new TensorRTInference(tensorrt_paths[1], gpuId);

    // ImageDecoderFirst_engine = new TensorRTInference(tensorrt_paths[2], gpuId);

    // ImageDecoderEnd_engine = new TensorRTInference(tensorrt_paths[3], gpuId);

    MemoryEncoder_engine = new TensorRTInference(tensorrt_paths[4], gpuId);

    ImageDecoder_engine = new TensorRTInference(tensorrt_paths[5], gpuId);

    if(TRACKTYPEBYSAM2::SingleTrack == this->trackType)
    {
        this->batch_size = 1;
        setparms_AllocateBatch_One();

        // warm up
        std::vector<cv::Rect> rectOut;
        cv::Rect rectInfoIn = cv::Rect(100, 100, 100, 100);
        std::vector<cv::Mat> images;
        for(size_t i = 0; i < 50; i++)
        {
            cv::Mat image_temp = cv::Mat::zeros(cv::Size(1920, 1080), CV_8UC3);
            images.push_back(image_temp);
        }

        sam2Process(images, rectInfoIn, rectOut);

        //clear old 记忆帧
        this->infer_status.current_frame = 0;
        this->infer_status.obj_ptr_first.clear();
        this->infer_status.status_first.clear();
        this->infer_status.status_recent.clear();
        this->infer_status.obj_ptr_recent.clear();
        this->infer_status.last_memoryFeature.clear();
        this->infer_status.last_objPtr.clear();
        this->parms.clear();
        this->LastRect = cv::Rect();

        is_initialized.store(true);
    }
    else if (TRACKTYPEBYSAM2::MultiTrack == this->trackType)
    {
        // batch_size 和内存分配推迟到 setparms() 时执行
        is_initialized.store(true);
    }

    return true;
}

bool Sam2Singleton::inference(cv::Mat &image){
    #ifdef ALL_TIMEDEBUG
    auto startall = std::chrono::high_resolution_clock::now();
    #endif

    this->LastRect = cv::Rect();
    if (image.empty() || !is_inited) 
    {
        return "image can not empyt!";
    }

    this->ori_img = &image;

    try {
        this->preprocess(image); //
     }catch (const std::exception& e) {
        std::cerr << "Image preprocess failed!" << std::endl;
        return "Image preprocess failed!";
    }

    #ifdef ALL_TIMEDEBUG
    auto enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endall1 for preprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << " ms" << std::endl;
    #endif

    //step1. image encoder
    //[1,3,1024,1024]
    // pix_feat,high_res_feat0,high_res_feat1,vision_feats,vision_pos_embed
    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<int> batch_size_data(this->batch_size, 2);
    ImageEndocer_engine->infer_ImageEncoder(gpu_buffers[4].data, batch_size_data, outputHostData, this->trackType);
    std::vector<Ort::Value> img_encoder_out;
    createTensorImgEncoder(img_encoder_out);

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endall2 for ImageEncoder_engine inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    //step2. mem_attention
    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<Ort::Value> mem_attention_out;
    this->mem_attention_infer(img_encoder_out, mem_attention_out);

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endall3 for memory_attenion inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif

    #ifdef TIME_DEBUG
    auto startMove = std::chrono::high_resolution_clock::now();
    #endif
    //如果last info 的置信度iou_mean大于0.5 那么我们把last info 放进history info
    if(0 != infer_status.last_memoryFeature.size())
    {
        std::vector<float> iou_tmp = infer_status.last_memoryFeature.at(0).iou;
        int idx_batchRecent = -1;
        for(size_t idx_batch = 0; idx_batch < this->batch_size; idx_batch++)
        {
            if(iou_tmp[idx_batch] >= 0.4)
            {
                //将last info放入history info
                //1. last_memoryFeature to status_recent
                // temp.maskmem_features.push_back(std::move(mem_encoder_out[0]));    //

                // temp.maskmem_pos_enc.push_back(std::move(mem_encoder_out[1]));
                // temp.temporal_code.push_back(std::move(mem_encoder_out[2]));
                //      maskmem_features  [1,64,64,64]
                //      maskmem_pos_enc   [4096,1,64]
                //      temporal_code     [7,1,1,64]
                SubStatus subStatu;
                subStatu.data_holder_maskmem_features = std::make_unique<std::vector<float>>(this->batch_size * 64 * 64 * 64);
                const float* ptr_maskmem_features = infer_status.last_memoryFeature.at(0).maskmem_features[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_maskmem_features->data(), ptr_maskmem_features, this->batch_size * 64 * 64 * 64 * sizeof(float));
                std::vector<int64_t> maskmem_features_tmp_shape = {this->batch_size, 64, 64, 64};
                Ort::Value maskmem_features_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_maskmem_features->data(),
                    subStatu.data_holder_maskmem_features->size(),
                    maskmem_features_tmp_shape.data(),
                    maskmem_features_tmp_shape.size()
                );
                subStatu.maskmem_features.push_back(std::move(maskmem_features_tmp_tensor));

                subStatu.data_holder_maskmem_pos_enc = std::make_unique<std::vector<float>>(4096 * this->batch_size * 64);
                const float* ptr_maskmem_pos_enc = infer_status.last_memoryFeature.at(0).maskmem_pos_enc[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_maskmem_pos_enc->data(), ptr_maskmem_pos_enc, this->batch_size * 64 * 4096 * sizeof(float));
                std::vector<int64_t> maskmem_pos_enc_tmp_shape = {4096, this->batch_size, 64};
                Ort::Value maskmem_pos_enc_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_maskmem_pos_enc->data(),
                    subStatu.data_holder_maskmem_pos_enc->size(),
                    maskmem_pos_enc_tmp_shape.data(),
                    maskmem_pos_enc_tmp_shape.size()
                );
                subStatu.maskmem_pos_enc.push_back(std::move(maskmem_pos_enc_tmp_tensor));

                subStatu.data_holder_temporal_code = std::make_unique<std::vector<float>>(7 * 64);
                const float* ptr_temporal_code = infer_status.last_memoryFeature.at(0).temporal_code[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_temporal_code->data(), ptr_temporal_code, 7 * 64 * sizeof(float));
                std::vector<int64_t> temporal_code_tmp_shape = {7, 1, 1, 64};
                Ort::Value temporal_code_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_temporal_code->data(),
                    subStatu.data_holder_temporal_code->size(),
                    temporal_code_tmp_shape.data(),
                    temporal_code_tmp_shape.size()
                );
                subStatu.temporal_code.push_back(std::move(temporal_code_tmp_tensor));


                infer_status.status_recent[idx_batch].push(std::move(subStatu));    //todo 这里是有问题的 已经被move走了

                //2. last_objPtr to obj_ptr_recent
                //创建一个ort::Value变量 copy obj_ptr的数据 然后放入到obj_ptr_recent[idx_batch]中
                Obj_Ptr_Status obj_tmp;
                obj_tmp.iou = iou_tmp;
                obj_tmp.data_holder = std::make_unique<std::vector<float>>(this->batch_size * 256);
                
                //copy infer_status.last_objPtr.at(0)
                const float* ptr_obj_ptr_ori = infer_status.last_objPtr.at(0).obj_ptr[0].GetTensorData<float>();
                std::memcpy(obj_tmp.data_holder->data(), ptr_obj_ptr_ori, this->batch_size * 256 * sizeof(float));

                std::vector<int64_t> obj_tmp_shape = {this->batch_size, 256};
                Ort::Value obj_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    obj_tmp.data_holder->data(),
                    obj_tmp.data_holder->size(),
                    obj_tmp_shape.data(),
                    obj_tmp_shape.size()
                );

                obj_tmp.obj_ptr.push_back(std::move(obj_tmp_tensor));
                infer_status.obj_ptr_recent[idx_batch].push(std::move(obj_tmp));
            }
        }
    }

    #ifdef TIME_DEBUG
    auto endMove = std::chrono::high_resolution_clock::now();
    auto durationMove = std::chrono::duration_cast<std::chrono::milliseconds>(endMove - startMove).count();
    std::cout << ", durationMove = " << durationMove << "ms" << std::endl;
    #endif

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endallr for durationMove inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    //*****************************img_decoder**********************************
    mem_attention_out.push_back(std::move(img_encoder_out[1])); // high_res_feat0
    mem_attention_out.push_back(std::move(img_encoder_out[2])); // high_res_feat1


    std::vector<float> point_val;
    std::vector<int> point_labels;
    creatPointInput(point_val, point_labels);

    // ImageDecoderFirst_engine->infer_ImageDecoderFirst(point_val, point_labels, mem_attention_out, outputDataForImageDecoder_First);
    // std::vector<Ort::Value> img_Startdeco;
    // creatImageDecoder_First(img_Startdeco);

    // #ifdef TIME_DEBUG
    // auto startKF = std::chrono::high_resolution_clock::now();
    // #endif

    sKfOut result_kf;    //TODO yanwendou 这里存在大量数据拷贝情况 需要分析内存使用
    // this->img_decoder_KF(img_Startdeco, result_kf);

    // #ifdef TIME_DEBUG
    // auto endKF = std::chrono::high_resolution_clock::now();
    // auto durationKF = std::chrono::duration_cast<std::chrono::milliseconds>(endKF - startKF).count();
    // std::cout << ", durationKF = " << durationKF << "ms" << std::endl;
    // #endif

    // // 存储推理状态
    // SubStatus temp;
    // temp.iou = result_kf.best_iou;
    // temp.obj_score = result_kf.obj_score;
    // //result_kf
    // /* 
    //     float best_iou;
    //     float obj_score;
    //     Ort::Value low_res_masks;
    //     Ort::Value high_res_masks;
    //     Ort::Value sam_output_token;
    //     Ort::Value object_score_logits;
    // */
    // const float* ptr_object_score_logits = result_kf.object_score_logits[0].GetTensorData<float>();
    // float value_object_score_logits = ptr_object_score_logits[0];
    // std::vector<int64_t> shape = {1, 1};  // 张量形状为 {1, 1}
    // Ort::Value tensor_object_score_logits = Ort::Value::CreateTensor<float>(
    //     memory_info,
    //     &value_object_score_logits, 
    //     1, 
    //     shape.data(), 
    //     shape.size()
    // );

    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif

    std::vector<Ort::Value> img_decoder_out;
    // ImageDecoderEnd_engine->infer_ImageDecoderSecond(result_kf, outputDataForImageDecoder_Second);
    std::vector<void *>inputStr;
    inputStr.push_back(mem_attention_engine_mem7_obj16->outputData[0]);
    inputStr.push_back(ImageEndocer_engine->outputData[1]);
    inputStr.push_back(ImageEndocer_engine->outputData[2]);
    ImageDecoder_engine->infer_ImageDecoder(point_val, point_labels, mem_attention_out, outputDataForImageDecoder_Second, inputStr);

    result_kf.best_iou = get_best_iou();    //取best iou
    result_kf.obj_score = 1;

    creatTensorImageDecoder_Second(img_decoder_out);

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endallr for img_decoder_out inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    //by yanwendou, 优化decoder
    //(1) 将多个multimask全部放出来
    //(2) 根据卡尔曼滤波器找其中最优的那个


    //***********************************************************************
    //更新last info
    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif 

    Obj_Ptr_Status objptrStatusTmp;
    objptrStatusTmp.obj_ptr.push_back(std::move(img_decoder_out[0]));

    objptrStatusTmp.iou = result_kf.best_iou;
    objptrStatusTmp.obj_score = result_kf.obj_score;

    SubStatus temp;
    temp.iou = result_kf.best_iou;
    temp.obj_score = result_kf.obj_score;

    if(infer_status.current_frame == 0)[[unlikely]]{
        infer_status.obj_ptr_first.push_back(std::move(objptrStatusTmp));    //conditioned frame
        std::vector<float> iou_tmp = infer_status.obj_ptr_first[0].iou;
            {
                Obj_Ptr_Status obj_tmp;
                obj_tmp.iou = iou_tmp;
                obj_tmp.data_holder = std::make_unique<std::vector<float>>(this->batch_size * 256);
                
                //copy infer_status.last_objPtr.at(0)
                const float* ptr_obj_ptr_ori = infer_status.obj_ptr_first[0].obj_ptr[0].GetTensorData<float>();
                std::memcpy(obj_tmp.data_holder->data(), ptr_obj_ptr_ori, this->batch_size * 256 * sizeof(float));

                std::vector<int64_t> obj_tmp_shape = {this->batch_size, 256};
                Ort::Value obj_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    obj_tmp.data_holder->data(),
                    obj_tmp.data_holder->size(),
                    obj_tmp_shape.data(),
                    obj_tmp_shape.size()
                );

                obj_tmp.obj_ptr.push_back(std::move(obj_tmp_tensor));
                infer_status.obj_ptr_first.clear();
                infer_status.obj_ptr_first.push_back(std::move(obj_tmp));
            }

            for(int i = 0; i < 14; i++)
            {
                for(int j = 0; j < this->batch_size; j ++)
                {
                    Obj_Ptr_Status obj_tmp;
                    obj_tmp.iou = iou_tmp;
                    obj_tmp.data_holder = std::make_unique<std::vector<float>>(this->batch_size * 256);
                    
                    //copy infer_status.last_objPtr.at(0)
                    const float* ptr_obj_ptr_ori = infer_status.obj_ptr_first[0].obj_ptr[0].GetTensorData<float>();
                    std::memcpy(obj_tmp.data_holder->data(), ptr_obj_ptr_ori, this->batch_size * 256 * sizeof(float));

                    std::vector<int64_t> obj_tmp_shape = {this->batch_size, 256};
                    Ort::Value obj_tmp_tensor = Ort::Value::CreateTensor<float>(
                        memory_info,
                        obj_tmp.data_holder->data(),
                        obj_tmp.data_holder->size(),
                        obj_tmp_shape.data(),
                        obj_tmp_shape.size()
                    );

                    obj_tmp.obj_ptr.push_back(std::move(obj_tmp_tensor));
                    infer_status.obj_ptr_recent[j].push(std::move(obj_tmp));
                }
            }

            {
                Obj_Ptr_Status obj_tmp;
                obj_tmp.iou = iou_tmp;
                obj_tmp.data_holder = std::make_unique<std::vector<float>>(this->batch_size * 256);
                
                //copy infer_status.last_objPtr.at(0)
                const float* ptr_obj_ptr_ori = infer_status.obj_ptr_first[0].obj_ptr[0].GetTensorData<float>();
                std::memcpy(obj_tmp.data_holder->data(), ptr_obj_ptr_ori, this->batch_size * 256 * sizeof(float));

                std::vector<int64_t> obj_tmp_shape = {this->batch_size, 256};
                Ort::Value obj_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    obj_tmp.data_holder->data(),
                    obj_tmp.data_holder->size(),
                    obj_tmp_shape.data(),
                    obj_tmp_shape.size()
                );

                obj_tmp.obj_ptr.push_back(std::move(obj_tmp_tensor));
                infer_status.last_objPtr.push(std::move(obj_tmp));
            }
    }else{
        infer_status.last_objPtr.push(std::move(objptrStatusTmp));
    }

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endall5 for objmove time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif

    //*******************************mem_encoder******************************
    std::vector<Ort::Value> mem_encoder_in;
    mem_encoder_in.push_back(std::move(img_decoder_out[1])); //mask_for_mem
    mem_encoder_in.push_back(std::move(img_encoder_out[0])); //pix_feat
    //mem_encoder_in.push_back(std::move(tensor_object_score_logits));    //2.0版本就注释 2.1版本就打开

    std::vector<void *>mem_encoder_inStr;
    mem_encoder_inStr.push_back(ImageDecoder_engine->outputData[1]);
    mem_encoder_inStr.push_back(ImageEndocer_engine->outputData[0]);

    MemoryEncoder_engine->infer_MemoryDecoder(mem_encoder_in, outputDataForMmeEncoder, mem_encoder_inStr);
    std::vector<Ort::Value> mem_encoder_out;
    creatTensorMemEncoder(mem_encoder_out);

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endall6 for mem_encoder time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    //***************************************************************
    // 存储推理状态

    // 假设 mem_encoder_out 是一个 std::vector<Ort::Value>，并且 temp 是 SubStatus 类型的对象
    temp.maskmem_features.push_back(std::move(mem_encoder_out[0]));
    temp.maskmem_pos_enc.push_back(std::move(mem_encoder_out[1]));
    temp.temporal_code.push_back(std::move(mem_encoder_out[2]));
    
    #ifdef ALL_TIMEDEBUG
    startall = std::chrono::high_resolution_clock::now();
    #endif

    //更新last info
    if(infer_status.current_frame == 0)[[unlikely]]{
            infer_status.status_first.push_back(std::move(temp));    //conditioned frame
            {
                SubStatus subStatu;
                subStatu.iou = infer_status.status_first[0].iou;
                subStatu.data_holder_maskmem_features = std::make_unique<std::vector<float>>(this->batch_size * 64 * 64 * 64);
                const float* ptr_maskmem_features = infer_status.status_first[0].maskmem_features[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_maskmem_features->data(), ptr_maskmem_features, this->batch_size * 64 * 64 * 64 * sizeof(float));
                std::vector<int64_t> maskmem_features_tmp_shape = {this->batch_size, 64, 64, 64};
                Ort::Value maskmem_features_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_maskmem_features->data(),
                    subStatu.data_holder_maskmem_features->size(),
                    maskmem_features_tmp_shape.data(),
                    maskmem_features_tmp_shape.size()
                );
                subStatu.maskmem_features.push_back(std::move(maskmem_features_tmp_tensor));

                subStatu.data_holder_maskmem_pos_enc = std::make_unique<std::vector<float>>(4096 * this->batch_size * 64);
                const float* ptr_maskmem_pos_enc = infer_status.status_first[0].maskmem_pos_enc[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_maskmem_pos_enc->data(), ptr_maskmem_pos_enc, this->batch_size * 64 * 4096 * sizeof(float));
                std::vector<int64_t> maskmem_pos_enc_tmp_shape = {4096, this->batch_size, 64};
                Ort::Value maskmem_pos_enc_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_maskmem_pos_enc->data(),
                    subStatu.data_holder_maskmem_pos_enc->size(),
                    maskmem_pos_enc_tmp_shape.data(),
                    maskmem_pos_enc_tmp_shape.size()
                );
                subStatu.maskmem_pos_enc.push_back(std::move(maskmem_pos_enc_tmp_tensor));

                subStatu.data_holder_temporal_code = std::make_unique<std::vector<float>>(7 * 64);
                const float* ptr_temporal_code = infer_status.status_first[0].temporal_code[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_temporal_code->data(), ptr_temporal_code, 7 * 64 * sizeof(float));
                std::vector<int64_t> temporal_code_tmp_shape = {7, 1, 1, 64};
                Ort::Value temporal_code_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_temporal_code->data(),
                    subStatu.data_holder_temporal_code->size(),
                    temporal_code_tmp_shape.data(),
                    temporal_code_tmp_shape.size()
                );
                subStatu.temporal_code.push_back(std::move(temporal_code_tmp_tensor));
                infer_status.status_first.clear();
                infer_status.status_first.push_back(std::move(subStatu));
            }

            for(int i = 0; i < 5; i++)
            {
                for(int j = 0; j < this->batch_size; j ++)
                {
                    SubStatus subStatu;
                    subStatu.iou = infer_status.status_first[0].iou;
                    subStatu.data_holder_maskmem_features = std::make_unique<std::vector<float>>(this->batch_size * 64 * 64 * 64);
                    const float* ptr_maskmem_features = infer_status.status_first[0].maskmem_features[0].GetTensorData<float>();
                    std::memcpy(subStatu.data_holder_maskmem_features->data(), ptr_maskmem_features, this->batch_size * 64 * 64 * 64 * sizeof(float));
                    std::vector<int64_t> maskmem_features_tmp_shape = {this->batch_size, 64, 64, 64};
                    Ort::Value maskmem_features_tmp_tensor = Ort::Value::CreateTensor<float>(
                        memory_info,
                        subStatu.data_holder_maskmem_features->data(),
                        subStatu.data_holder_maskmem_features->size(),
                        maskmem_features_tmp_shape.data(),
                        maskmem_features_tmp_shape.size()
                    );
                    subStatu.maskmem_features.push_back(std::move(maskmem_features_tmp_tensor));

                    subStatu.data_holder_maskmem_pos_enc = std::make_unique<std::vector<float>>(4096 * this->batch_size * 64);
                    const float* ptr_maskmem_pos_enc = infer_status.status_first[0].maskmem_pos_enc[0].GetTensorData<float>();
                    std::memcpy(subStatu.data_holder_maskmem_pos_enc->data(), ptr_maskmem_pos_enc, this->batch_size * 64 * 4096 * sizeof(float));
                    std::vector<int64_t> maskmem_pos_enc_tmp_shape = {4096, this->batch_size, 64};
                    Ort::Value maskmem_pos_enc_tmp_tensor = Ort::Value::CreateTensor<float>(
                        memory_info,
                        subStatu.data_holder_maskmem_pos_enc->data(),
                        subStatu.data_holder_maskmem_pos_enc->size(),
                        maskmem_pos_enc_tmp_shape.data(),
                        maskmem_pos_enc_tmp_shape.size()
                    );
                    subStatu.maskmem_pos_enc.push_back(std::move(maskmem_pos_enc_tmp_tensor));

                    subStatu.data_holder_temporal_code = std::make_unique<std::vector<float>>(7 * 64);
                    const float* ptr_temporal_code = infer_status.status_first[0].temporal_code[0].GetTensorData<float>();
                    std::memcpy(subStatu.data_holder_temporal_code->data(), ptr_temporal_code, 7 * 64 * sizeof(float));
                    std::vector<int64_t> temporal_code_tmp_shape = {7, 1, 1, 64};
                    Ort::Value temporal_code_tmp_tensor = Ort::Value::CreateTensor<float>(
                        memory_info,
                        subStatu.data_holder_temporal_code->data(),
                        subStatu.data_holder_temporal_code->size(),
                        temporal_code_tmp_shape.data(),
                        temporal_code_tmp_shape.size()
                    );
                    subStatu.temporal_code.push_back(std::move(temporal_code_tmp_tensor));
        
                    infer_status.status_recent[j].push(std::move(subStatu));
                }
            }


            {
                SubStatus subStatu;
                subStatu.iou = infer_status.status_first[0].iou;
                subStatu.data_holder_maskmem_features = std::make_unique<std::vector<float>>(this->batch_size * 64 * 64 * 64);
                const float* ptr_maskmem_features = infer_status.status_first[0].maskmem_features[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_maskmem_features->data(), ptr_maskmem_features, this->batch_size * 64 * 64 * 64 * sizeof(float));
                std::vector<int64_t> maskmem_features_tmp_shape = {this->batch_size, 64, 64, 64};
                Ort::Value maskmem_features_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_maskmem_features->data(),
                    subStatu.data_holder_maskmem_features->size(),
                    maskmem_features_tmp_shape.data(),
                    maskmem_features_tmp_shape.size()
                );
                subStatu.maskmem_features.push_back(std::move(maskmem_features_tmp_tensor));

                subStatu.data_holder_maskmem_pos_enc = std::make_unique<std::vector<float>>(4096 * this->batch_size * 64);
                const float* ptr_maskmem_pos_enc = infer_status.status_first[0].maskmem_pos_enc[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_maskmem_pos_enc->data(), ptr_maskmem_pos_enc, this->batch_size * 64 * 4096 * sizeof(float));
                std::vector<int64_t> maskmem_pos_enc_tmp_shape = {4096, this->batch_size, 64};
                Ort::Value maskmem_pos_enc_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_maskmem_pos_enc->data(),
                    subStatu.data_holder_maskmem_pos_enc->size(),
                    maskmem_pos_enc_tmp_shape.data(),
                    maskmem_pos_enc_tmp_shape.size()
                );
                subStatu.maskmem_pos_enc.push_back(std::move(maskmem_pos_enc_tmp_tensor));

                subStatu.data_holder_temporal_code = std::make_unique<std::vector<float>>(7 * 64);
                const float* ptr_temporal_code = infer_status.status_first[0].temporal_code[0].GetTensorData<float>();
                std::memcpy(subStatu.data_holder_temporal_code->data(), ptr_temporal_code, 7 * 64 * sizeof(float));
                std::vector<int64_t> temporal_code_tmp_shape = {7, 1, 1, 64};
                Ort::Value temporal_code_tmp_tensor = Ort::Value::CreateTensor<float>(
                    memory_info,
                    subStatu.data_holder_temporal_code->data(),
                    subStatu.data_holder_temporal_code->size(),
                    temporal_code_tmp_shape.data(),
                    temporal_code_tmp_shape.size()
                );
                subStatu.temporal_code.push_back(std::move(temporal_code_tmp_tensor));
        
                infer_status.last_memoryFeature.push(std::move(subStatu));
            }
    }
    else
    {
        {
            SubStatus subStatu;
            subStatu.iou = temp.iou;
            subStatu.data_holder_maskmem_features = std::make_unique<std::vector<float>>(this->batch_size * 64 * 64 * 64);
            const float* ptr_maskmem_features = temp.maskmem_features[0].GetTensorData<float>();
            std::memcpy(subStatu.data_holder_maskmem_features->data(), ptr_maskmem_features, this->batch_size * 64 * 64 * 64 * sizeof(float));
            std::vector<int64_t> maskmem_features_tmp_shape = {this->batch_size, 64, 64, 64};
            Ort::Value maskmem_features_tmp_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                subStatu.data_holder_maskmem_features->data(),
                subStatu.data_holder_maskmem_features->size(),
                maskmem_features_tmp_shape.data(),
                maskmem_features_tmp_shape.size()
            );
            subStatu.maskmem_features.push_back(std::move(maskmem_features_tmp_tensor));

            subStatu.data_holder_maskmem_pos_enc = std::make_unique<std::vector<float>>(4096 * this->batch_size * 64);
            const float* ptr_maskmem_pos_enc = temp.maskmem_pos_enc[0].GetTensorData<float>();
            std::memcpy(subStatu.data_holder_maskmem_pos_enc->data(), ptr_maskmem_pos_enc, this->batch_size * 64 * 4096 * sizeof(float));
            std::vector<int64_t> maskmem_pos_enc_tmp_shape = {4096, this->batch_size, 64};
            Ort::Value maskmem_pos_enc_tmp_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                subStatu.data_holder_maskmem_pos_enc->data(),
                subStatu.data_holder_maskmem_pos_enc->size(),
                maskmem_pos_enc_tmp_shape.data(),
                maskmem_pos_enc_tmp_shape.size()
            );
            subStatu.maskmem_pos_enc.push_back(std::move(maskmem_pos_enc_tmp_tensor));

            subStatu.data_holder_temporal_code = std::make_unique<std::vector<float>>(7 * 64);
            const float* ptr_temporal_code = temp.temporal_code[0].GetTensorData<float>();
            std::memcpy(subStatu.data_holder_temporal_code->data(), ptr_temporal_code, 7 * 64 * sizeof(float));
            std::vector<int64_t> temporal_code_tmp_shape = {7, 1, 1, 64};
            Ort::Value temporal_code_tmp_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                subStatu.data_holder_temporal_code->data(),
                subStatu.data_holder_temporal_code->size(),
                temporal_code_tmp_shape.data(),
                temporal_code_tmp_shape.size()
            );
            subStatu.temporal_code.push_back(std::move(temporal_code_tmp_tensor));
    
            infer_status.last_memoryFeature.push(std::move(subStatu));
        }
    }

    #ifdef ALL_TIMEDEBUG
    enddall = std::chrono::high_resolution_clock::now();
    std::cout << "endall7 for featuremove time: " << std::chrono::duration_cast<std::chrono::milliseconds>(enddall - startall).count() << "ms" << std::endl;
    #endif

    //***************************************************************

    // 后处理
    std::vector<Ort::Value> output_tensors;
    output_tensors.push_back(std::move(img_decoder_out[2]));   //pred_mask
    try {
            #ifdef ALL_TIMEDEBUG
            auto start = std::chrono::high_resolution_clock::now();
            #endif

            this->postprocess(output_tensors);

            #ifdef ALL_TIMEDEBUG
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            std::cout << "endall8 time: " << duration << " ms" << std::endl;
            #endif
    }catch (const std::exception& e) {
        return "tensor postprocess failed!!";
    }
    this->infer_status.current_frame++;
    return true;
}
// input [1,3,1024,1024]
// output: 
//      pix_feat        [1,256,64,64]
//      high_res_feat0  [1,32,256,256]
//      high_res_feat1  [1,64,128,128]
//      vision_feats    [1,256,64,64]
//      vision_pos_embed [4096,1,256]

// input: 
//    pix_feat          [1,256,64,64]
//    high_res_feat0    [1,32,256,256]
//    high_res_feat1    [1,64,128,128]
//    vision_feats      [1,256,64,64]
//    vision_pos_embed  [4096, 1, 256]
// out:
//    image_embed   [1,256,64,64]
void Sam2Singleton::mem_attention_infer(std::vector<Ort::Value>&img_encoder_out, std::vector<Ort::Value>& mem_attention_out){
    if(infer_status.current_frame == 0) [[unlikely]]{
        mem_attention_out.push_back(std::move(img_encoder_out[3]));

        // 第0帧 特殊处理
        std::memcpy(tensorrt_mem_attention_data.data(), mem_attention_out[0].GetTensorData<float>(), this->batch_size * 256 * 64 * 64 * sizeof(float));
        cudaMemcpy(mem_attention_engine_mem7_obj16->outputData[0],
                   tensorrt_mem_attention_data.data(), this->batch_size * 256 * 64 * 64 * sizeof(float), cudaMemcpyHostToDevice);
        return;
    }
    //*******************************************************************************
    //创建输入数据 current_vision_feat，current_vision_pos_embed，memory_0，memory_1，memory_pos_embed
    std::vector<Ort::Value> input_tensor; // 5
    input_tensor.push_back(std::move(img_encoder_out[3])); //current_vision_feat
    input_tensor.push_back(std::move(img_encoder_out[4])); //current_vision_pos_embed




    //step1, 处理每个batch_size的objPtr特征
    /*
        思路如下：
            总:每个batch_size 的高置信度的记忆特征 不可能总是由同一帧得到
                    比如说目标0的高置信度特征在 frame_idx:1得到，是帧1的输出的记忆特征中的第一个batch_size的特征
                    而目标1的高置信度特征frame_idx:0得到，是帧0的输出的记忆特征中的第一个batch_size的特征

                我们给每个目标 单独分配objPtr Recent池，每个batch都取最近邻的高置信度特征 拼在一起作为本帧推理的记忆输入

            有个问题：当其中一个batch的objPtr Recent不足5个怎么办，其他的足5个，这个时候不能置0 0是有含义的，不能随便置0进去
            这个时候 就把最近的 放到最开始占比会重些  缺的 按照最多的batch的objrECNT数量 去使用其他obj的帧的obj的batch 作为自己的 
    */

    // 需要构造出 memory 和 memory_pos_embed
    // memory是由obj_ptr_first，obj_ptr_recent和status_recent.maskmem_features 构造出的
    // 中间的
    // 比较batch_size个 recent 中 最大的那个，遍历这个
    int max_RecentSizeInbatch = 0;
    int index_maxRecentSize   = 0;
    for(size_t idx_maxRecSize = 0; idx_maxRecSize < infer_status.obj_ptr_recent.size(); idx_maxRecSize++)
    {
        if(infer_status.obj_ptr_recent[idx_maxRecSize].size() > max_RecentSizeInbatch)
        {
            max_RecentSizeInbatch = infer_status.obj_ptr_recent[idx_maxRecSize].size();
            index_maxRecentSize = idx_maxRecSize;
        }
    }
    int obj_buffer_size = 1 + max_RecentSizeInbatch +  infer_status.last_objPtr.size();    //1+0,1+1,1+2,...,1+15

    std::vector<int64_t> dimensions_0{(int64_t)obj_buffer_size, this->batch_size, 256}; // [y,256*batch_size]

    // std::vector<float> obj_ptrs(obj_buffer_size * 256 * this->batch_size); // first+recent // 16*256 * batch_size

    const float* tensor_data = infer_status.obj_ptr_first[0].obj_ptr[0].GetTensorData<float>();
    // std::copy_n(tensor_data, 256*this->batch_size, std::begin(obj_ptrs));
    memcpy(obj_ptrs.data(), tensor_data, sizeof(float) * 256 * batch_size);

    // for(size_t i = 0;i<infer_status.obj_ptr_recent.size();i++){
    //     auto& temp_tensor = infer_status.obj_ptr_recent.at(i).obj_ptr[0];
    //     tensor_data = temp_tensor.GetTensorData<float>();
    //     std::copy_n(tensor_data, 256*this->batch_size, std::begin(obj_ptrs)+256*this->batch_size*(i+1));
    // }
    for(size_t idx_rect = 0; idx_rect < max_RecentSizeInbatch; idx_rect++)
    {
        for(size_t idx_batch = 0; idx_batch < this->batch_size; idx_batch++)
        {
            if(idx_rect >= infer_status.obj_ptr_recent[idx_batch].size())
            {
                auto& temp_tensor = infer_status.obj_ptr_recent[index_maxRecentSize].at(idx_rect).obj_ptr[0];
                tensor_data = temp_tensor.GetTensorData<float>();
                // std::copy_n(tensor_data + idx_batch * 256, 256, std::begin(obj_ptrs)+256*this->batch_size*(idx_rect+1) + idx_batch * 256);
                std::memcpy(
                    obj_ptrs.data() + 256 * batch_size * (idx_rect + 1) + idx_batch * 256,
                    tensor_data + idx_batch * 256,
                    256 * sizeof(float)
                );
                //todo 应该是copy别人的mem 权重低些
            }
            else
            {
                auto& temp_tensor = infer_status.obj_ptr_recent[idx_batch].at(idx_rect).obj_ptr[0];
                tensor_data = temp_tensor.GetTensorData<float>();
                // std::copy_n(tensor_data + idx_batch * 256, 256, std::begin(obj_ptrs)+256*this->batch_size*(idx_rect+1) + idx_batch * 256);
                std::memcpy(
                    obj_ptrs.data() + 256 * this->batch_size * (idx_rect + 1) + idx_batch * 256,
                    tensor_data + idx_batch * 256,
                    256 * sizeof(float)
                );
            }
        }
    }

    if(0 != infer_status.last_objPtr.size())
    {
        auto& temp_tensor = infer_status.last_objPtr.at(0).obj_ptr[0];
        tensor_data = temp_tensor.GetTensorData<float>();
        std::memcpy(
            obj_ptrs.data() + 256 * this->batch_size * (obj_buffer_size - 1),
            tensor_data,
            256 * this->batch_size * sizeof(float)
        );
    }

    auto memory_0 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    obj_ptrs.data(),
                    obj_ptrs.size(),
                    dimensions_0.data(),
                    dimensions_0.size()
    );



    max_RecentSizeInbatch = 0;
    index_maxRecentSize   = 0;
    for(size_t idx_maxRecSize = 0; idx_maxRecSize < infer_status.status_recent.size(); idx_maxRecSize++)
    {
        if(infer_status.status_recent[idx_maxRecSize].size() > max_RecentSizeInbatch)
        {
            max_RecentSizeInbatch = infer_status.status_recent[idx_maxRecSize].size();
            index_maxRecentSize = idx_maxRecSize;
        }
    }
    size_t features_size = 1 + max_RecentSizeInbatch +  infer_status.last_memoryFeature.size(); // 1,2,3,...,7
    // std::vector<float> maskmem_features_(features_size*64*64*64*this->batch_size); //固定为7

    tensor_data = infer_status.status_first[0].maskmem_features[0].GetTensorData<float>();
    // std::copy_n(tensor_data, this->batch_size*64*64*64, std::begin(maskmem_features_));
    std::memcpy(maskmem_features_.data(), tensor_data, this->batch_size * 64 * 64 * 64 * sizeof(float));







    // for(size_t i = 0; i<infer_status.status_recent.size(); i++){
    //     auto& temp_tensor = this->infer_status.status_recent.at(i).maskmem_features;
    //     tensor_data = temp_tensor[0].GetTensorData<float>();
    //     std::copy_n(tensor_data, 64*64*64*this->batch_size, std::begin(maskmem_features_)+64*64*64*this->batch_size*(i+1));
    // }

    for(size_t idx_rect = 0; idx_rect < max_RecentSizeInbatch; idx_rect++)
    {
        for(size_t idx_batch = 0; idx_batch < this->batch_size; idx_batch++)
        {
            if(idx_rect >= this->infer_status.status_recent[idx_batch].size())
            {
                auto& temp_tensor = this->infer_status.status_recent[index_maxRecentSize].at(idx_rect).maskmem_features;
                tensor_data = temp_tensor[0].GetTensorData<float>();
                // std::copy_n(tensor_data + idx_batch*64*64*64, 64*64*64, 
                //             std::begin(maskmem_features_)+64*64*64*this->batch_size*(idx_rect+1) + idx_batch * 64*64*64);
                std::memcpy(
                    maskmem_features_.data() + 64 * 64 * 64 * this->batch_size * (idx_rect + 1) + idx_batch * 64 * 64 * 64,
                    tensor_data + idx_batch * 64 * 64 * 64,
                    64 * 64 * 64 * sizeof(float)
                );
                
            }
            else
            {
                auto& temp_tensor = this->infer_status.status_recent[idx_batch].at(idx_rect).maskmem_features;
                tensor_data = temp_tensor[0].GetTensorData<float>();
                // std::copy_n(tensor_data + idx_batch*64*64*64, 64*64*64, 
                //             std::begin(maskmem_features_)+64*64*64*this->batch_size*(idx_rect+1) + idx_batch * 64*64*64);
                std::memcpy(
                    maskmem_features_.data() + 64 * 64 * 64 * this->batch_size * (idx_rect + 1) + idx_batch * 64 * 64 * 64,
                    tensor_data + idx_batch * 64 * 64 * 64,
                    64 * 64 * 64 * sizeof(float)
                );
            }
        }
    }













    if(0 != infer_status.last_memoryFeature.size())
    {
        auto& temp_tensor = infer_status.last_memoryFeature.at(0).maskmem_features;
        tensor_data = temp_tensor[0].GetTensorData<float>();
        // std::copy_n(tensor_data, this->batch_size*64*64*64, std::begin(maskmem_features_)+64*64*64*this->batch_size*(features_size - 1));
        std::memcpy(
            maskmem_features_.data() + 64 * 64 * 64 * this->batch_size * (features_size - 1),
            tensor_data,
            this->batch_size * 64 * 64 * 64 * sizeof(float)
        );
    }

    std::vector<int64_t> dimensions_1{(int64_t)features_size,this->batch_size,64,64,64}; // [x,64,64,64]
    auto memory_1 = Ort::Value::CreateTensor<float>(
                    memory_info,
                    maskmem_features_.data(),
                    maskmem_features_.size(),
                    dimensions_1.data(),
                    dimensions_1.size()
                    );
    input_tensor.push_back(std::move(memory_0));
    input_tensor.push_back(std::move(memory_1));


















    //***********************************************************************
    // memory_pos_embed是由两部分组成的。
    auto& temp_time = (infer_status.current_frame == 1) 
    ? infer_status.status_first[0].temporal_code
    : infer_status.last_memoryFeature.at(0).temporal_code;


    const float* temporal_code_ = temp_time[0].GetTensorData<float>(); // [7,64]    //权重 所以倒序
    std::vector<const float*> temporal_code;
    for(int i = 6;i>=0;i--){
        auto temp = temporal_code_+i*64;
        temporal_code.push_back(temp);
    }

    size_t maskmem_buffer_size = features_size;
    size_t maskmem_pos_enc_size = (maskmem_buffer_size*4096+4*std::min(obj_buffer_size, 16))*this->batch_size*64;
    
    // std::vector<float> maskmem_pos_enc_(maskmem_pos_enc_size);
    
    // // a[] , b[4096,1,64], c[1,1,64]
    // auto tensor_add = [&](float* a,const float* b,const float* c){
    //     // b+c,结果保存到a
    //     for(int i =0;i<4096;i++){
    //         for(int j =0;j<64;j++){
    //             a[i*64+j] = b[i*64+j] + c[j];
    //         }
    //     }
    // };

    // a[] , b[4096, batch_size, 64], c[1, 1, 64]
    auto tensor_add = [&](float* a, const float* b, const float* c) {
        const int B = this->batch_size;
        const int stride_j = 64;
        const int stride_i = B * 64;
        
        #pragma omp parallel for
        for (int i = 0; i < 4096; ++i) {
            const float* b_ptr = b + i * stride_i;
            float* a_ptr = a + i * stride_i;
            for (int j = 0; j < B; ++j) {
                const float* b_ij = b_ptr + j * stride_j;
                float* a_ij = a_ptr + j * stride_j;     // 缓存了指针  避免每一次直接访问数组 加强cpu缓存命中率
                for (int k = 0; k < 64; ++k) {
                    a_ij[k] = b_ij[k] + c[k];
                }
            }
        }
    };

    // 第一部分：
    auto& temp_tensor = this->infer_status.status_first[0].maskmem_pos_enc;
    auto sub = temp_tensor[0].GetTensorData<float>();
    float* p = maskmem_pos_enc_.data();
    tensor_add(p,sub,temporal_code.at(0));



    // for(size_t j = 0; j<infer_status.status_recent.size(); j++){
    //     auto& temp_tensor = this->infer_status.status_recent.at(j).maskmem_pos_enc;
    //     auto sub = temp_tensor[0].GetTensorData<float>();//[4096,1,64]
    //     float* p = maskmem_pos_enc_.data() + (j + 1)*4096*64*this->batch_size;
    //     tensor_add(p,sub,temporal_code.at((j + 1))); // [4096,batch_size,64] + [1,batch_size,64] ->[4096,batch_size,64] + [4096,batch_size,64] ->[4096,batch_size,64]
    // }
    for(size_t idx_rect = 0; idx_rect < max_RecentSizeInbatch; idx_rect++)
    {
        for(size_t idx_batch = 0; idx_batch < this->batch_size; idx_batch++)
        {
            if(idx_rect >= infer_status.status_recent[idx_batch].size())
            {
                auto& temp_tensor = this->infer_status.status_recent[index_maxRecentSize].at(idx_rect).maskmem_pos_enc;
                auto sub = temp_tensor[0].GetTensorData<float>();//[4096,1,64]
                float* p = maskmem_pos_enc_.data() + (idx_rect + 1)*4096*64*this->batch_size + idx_batch*4096*64;
                tensor_add(p,sub + idx_batch*4096*64, temporal_code.at((idx_rect + 1))); 
            }
            else
            {
                auto& temp_tensor = this->infer_status.status_recent[idx_batch].at(idx_rect).maskmem_pos_enc;
                auto sub = temp_tensor[0].GetTensorData<float>();//[4096,1,64]
                float* p = maskmem_pos_enc_.data() + (idx_rect + 1)*4096*64*this->batch_size + idx_batch*4096*64;
                tensor_add(p,sub + idx_batch*4096*64, temporal_code.at((idx_rect + 1))); 
            }
        }
    }


















    if(infer_status.last_memoryFeature.size())
    {
        auto& temp_tensor = this->infer_status.last_memoryFeature.at(0).maskmem_pos_enc;
        sub = temp_tensor[0].GetTensorData<float>();
        p = maskmem_pos_enc_.data() + (maskmem_buffer_size - 1)*4096*64*this->batch_size;
        tensor_add(p,sub,temporal_code.at(maskmem_buffer_size - 1));
    }

    // 第二部分：
    // yanwendou 这个是4*std::min(infer_status.current_frame,16))*64
   std::fill_n(maskmem_pos_enc_.begin()+maskmem_buffer_size*4096*64*batch_size, maskmem_pos_enc_size - (maskmem_buffer_size*4096*64*batch_size), 0.0f);

    // [z,batch_size,64]
    std::vector<int64_t> dimensions_3{int64_t(maskmem_buffer_size*4096+4*std::min(obj_buffer_size,16)),this->batch_size,64};
    auto memory_pos_embed = Ort::Value::CreateTensor<float>(
                        memory_info,
                        maskmem_pos_enc_.data(),
                        maskmem_pos_enc_.size(),
                        dimensions_3.data(),
                        dimensions_3.size()
                        );
    input_tensor.push_back(std::move(memory_pos_embed));

    std::vector<int64_t> shape_batch_size = {this->batch_size};  // 形状为 [batch_size]
    std::vector<int> batch_size_data(this->batch_size, 2);  // 创建一个大小为 batch_size 的 vector，值都为 2
    Ort::Value batch_size_tensor = Ort::Value::CreateTensor<int>(memory_info,
                                                                batch_size_data.data(), 
                                                                batch_size_data.size(), 
                                                                shape_batch_size.data(), 
                                                                shape_batch_size.size());

    input_tensor.push_back(std::move(batch_size_tensor));

    // 提前创建全局image_embed的vector
    // todo 上面这一串 流程优化 + 零拷贝 + 页固定技术 待实现
    mem_attention_engine_mem7_obj16->MemAttentioninfer(input_tensor, tensorrt_mem_attention_data.data());

    // 根据infer数据 创建ort::Value push_back进mem_attention_out中
    // std::vector<float> tensorrt_mem_attention_data shape是{this->batch_size, 256, 64, 64}}

    std::vector<int64_t> shape_image_embed = {this->batch_size, 256, 64, 64};  // 形状为 [batch_size]

    Ort::Value image_embed_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                    tensorrt_mem_attention_data.data(),
                                                                    tensorrt_mem_attention_data.size(),
                                                                    shape_image_embed.data(), 
                                                                    shape_image_embed.size());

    mem_attention_out.push_back(std::move(image_embed_tensor));

    return;
}

// std::vector<Ort::Value>,std::string> Sam2Singleton::img_decoder_infer(std::vector<Ort::Value>&mem_attention_out)
// {
//     return "img_decoder_inferStart failed";
// }


    /*  out info
    0. img_Startdeco_out[0] object_score_logits  [1, 1]
    1. img_Startdeco_out[1] low_res_multimasks   [1, 1, 256, 256]
    2. img_Startdeco_out[2] high_res_multimasks  [1, 1, 1024, 1024]
    3. img_Startdeco_out[3] sam_output_tokens    [1, 1, 256]
    4. img_Startdeco_out[4] ious [1, 1]
    5. img_Startdeco_out[5] low_res_masks    [1, 1, 256, 256]
    6. img_Startdeco_out[6] high_res_masks   [1, 1, 1024, 1024]
    7. img_Startdeco_out[7] sam_output_token [1, 256]
    */



cv::Point if_kf_tl; // 卡尔曼预测框左上角
cv::Point if_kf_br; // 卡尔曼预测框右下角
void Sam2Singleton::img_decoder_KF(std::vector<Ort::Value>&decoderKfIn, sKfOut& KfResultout)
{
    /* 
    IN:
        0. img_Startdeco_out[0] object_score_logits  [1, 1]
        1. img_Startdeco_out[1] low_res_multimasks   [1, 3, 256, 256]
        2. img_Startdeco_out[2] high_res_multimasks  [1, 3, 1024, 1024]
        3. img_Startdeco_out[3] sam_output_tokens    [1, 3, 256]
        4. img_Startdeco_out[4] ious [1, 3]
        5. img_Startdeco_out[5] low_res_masks    [1, 1, 256, 256]
        6. img_Startdeco_out[6] high_res_masks   [1, 1, 1024, 1024]
        7. img_Startdeco_out[7] sam_output_token [1, 256]    
        8. img_Startdeco_out[8] 多余输出 切记不可用
    INNeed: 
        0: ious               [batch_size, 3]
        1: sam_output_tokens  [batch_size, 3, H1, W1]

    infer.state:
        0: stable_frames
        1: frame_cnt
        2: stable_frames_threshold
        3: stable_ious_threshold
        4: kf_score_weight

    OUT:
        0: sam_output_token
     */

    //0. 整理输入
    Ort::Value& ious_tensor = decoderKfIn[4];
    Ort::Value& sam_output_tokens_tensor = decoderKfIn[3];
    Ort::Value& low_res_multimasks_tensor = decoderKfIn[1];
    Ort::Value& high_res_multimasks_tensor = decoderKfIn[2];
    Ort::Value& sam_output_token_tensor_origin = decoderKfIn[7];
    Ort::Value& low_res_masks_tensor_origin = decoderKfIn[5];
    Ort::Value& high_res_masks_tensor_origin = decoderKfIn[6];

    const float* ptr_object_score_logits = decoderKfIn[0].GetTensorData<float>();
    const float* ptr_ious = ious_tensor.GetTensorData<float>();    //[this->batch_szie, 3]
    const float* ptr_low_res_multimasks = low_res_multimasks_tensor.GetTensorData<float>();
    const float* high_res_multimasks_data = high_res_multimasks_tensor.GetTensorData<float>();
    const float* ptr_low_res_masks = low_res_masks_tensor_origin.GetTensorData<float>();
    const float* ptr_high_res_masks = high_res_masks_tensor_origin.GetTensorData<float>();
    const float* ptr_sam_token = sam_output_token_tensor_origin.GetTensorData<float>();
    const float* ptr_sam_tokens = sam_output_tokens_tensor.GetTensorData<float>();

    int height = 1024;
    int width  = 1024;

    //1. 如果当前帧是0 为第0个目标分配kf tracker 并按当前位置初始化状态
    if(0 == this->infer_status.current_frame)    //true
    {
        //TODO 和原版在第一帧的处理上不一样 这里暂时按第0帧 根据提示框初始化卡尔曼滤波器 限制只能使用bbox 
        for(size_t i = 0; i < this->batch_size; i++)
        {
            vector<float> tlwh_;
            tlwh_.resize(4);
            tlwh_[0] = this->parms[i].prompt_box.x * 1024.0 / this->ori_img->cols;
            tlwh_[1] = this->parms[i].prompt_box.y * 1024.0 / this->ori_img->rows;
            tlwh_[2] = this->parms[i].prompt_box.width * 1024.0 / this->ori_img->cols;
            tlwh_[3] = this->parms[i].prompt_box.height * 1024.0 / this->ori_img->rows;

            STrackInSam2 kf_tracker(tlwh_, 1, 0);

            kf_tracker.activate(this->infer_status.current_frame);
            kf_tracker.stable_frames ++;
            tracked_stracks.push_back(kf_tracker);

            KfResultout.best_iou.push_back(ptr_ious[0]);
            ptr_ious += 3;
        }

        //KfResultout.best_iou;//todo 暂时禁用记忆帧优化 会影响多目标效果KfResultout.best_iou / this->batch_size;    //取mean置信度    会影响单目标的结果
        KfResultout.obj_score = 1;    //TODO yanwendou
        KfResultout.low_res_masks.push_back(std::move(decoderKfIn[5]));
        KfResultout.high_res_masks.push_back(std::move(decoderKfIn[6]));
        KfResultout.sam_output_token.push_back(std::move(decoderKfIn[7]));
        KfResultout.object_score_logits.push_back(std::move(decoderKfIn[0]));
    }

    //2. 如果当前帧非0    遍历所有目标的stable_frame, 如果不足 那么就更新状态    如果足了 那么就进行预测
    else
    {
        // 使用指定大小初始化，并将所有元素设为 0    具体原因见初始化时的含义说明
        low_res_masks_data.assign(low_res_masks_data.size(), 0.0f);
        high_res_masks_data.assign(high_res_masks_data.size(), 0.0f);
        sam_output_data.assign(sam_output_data.size(), 0.0f);
        for(size_t idx_batch= 0; idx_batch < this->batch_size; idx_batch++)
        {
            if(0 == tracked_stracks[idx_batch].stable_frames)
            {
                //重新初始化 或者删除原来的 新构造一个

                //step1.
                std::vector<float> ious_data;
                for (size_t idx_ious = 0; idx_ious < 3; ++idx_ious)
                {
                    ious_data.push_back(ptr_ious[idx_ious + idx_batch * 3]);
                }
                auto max_iou = std::max_element(ious_data.begin(), ious_data.end());
                int best_iou_ind = std::distance(ious_data.begin(), max_iou);

                //step2. 取low res mask, 进行地址偏移    low_res_masks_data [this->batch_size, 1, 256, 256]    ptr_low_res_multimasks [this->batch_size, 3, 256, 256]
                float* ptr_dst_low       = low_res_masks_data.data() + idx_batch * 256 * 256;
                const float* ptr_src_low = ptr_low_res_masks + idx_batch * 256 * 256;
                std::memcpy(ptr_dst_low, ptr_src_low, 256 * 256 * sizeof(float));

                //step3. 取high res mask, 进行地址偏移
                float* ptr_dst_high       = high_res_masks_data.data() + idx_batch * 1024 * 1024;
                const float* ptr_src_high = ptr_high_res_masks + idx_batch * 1024 * 1024;
                std::memcpy(ptr_dst_high, ptr_src_high, 1024 * 1024 * sizeof(float));

                //step4. 取sam token out, 进行地址偏移
                float* ptr_dst_samtoken   = sam_output_data.data() + idx_batch * 256;
                const float* ptr_src_samtoken = ptr_sam_token + idx_batch * 256;
                std::memcpy(ptr_dst_samtoken, ptr_src_samtoken, 256 * sizeof(float));

                //step5. 处理卡尔曼信息
                tracked_stracks[idx_batch].single_predict();
                std::vector<float> tlwh_;
                bool found_non_zero = found_tlwh(ptr_high_res_masks + idx_batch * 1024 * 1024, tlwh_);

                tracked_stracks[idx_batch].activate(this->infer_status.current_frame);
                tracked_stracks[idx_batch].stable_frames += 1;

                float a = ptr_object_score_logits[idx_batch];
                if(true != std::isnan(ious_data[best_iou_ind]))
                {
                    KfResultout.best_iou.push_back(ious_data[best_iou_ind]);
                }
                else
                {
                    KfResultout.best_iou.push_back(0);
                }
            }
            else if(tracked_stracks[idx_batch].stable_frames < this->infer_status.stable_frames_threshold)
            {
                //step1.
                std::vector<float> ious_data;
                for (size_t idx_ious = 0; idx_ious < 3; ++idx_ious)
                {
                    ious_data.push_back(ptr_ious[idx_ious + idx_batch * 3]);
                }
                auto max_iou = std::max_element(ious_data.begin(), ious_data.end());
                int best_iou_ind = std::distance(ious_data.begin(), max_iou);

                //step2. 取low res mask, 进行地址偏移    low_res_masks_data [this->batch_size, 1, 256, 256]    ptr_low_res_multimasks [this->batch_size, 3, 256, 256]
                float* ptr_dst_low       = low_res_masks_data.data() + idx_batch * 256 * 256;
                const float* ptr_src_low = ptr_low_res_masks + idx_batch * 256 * 256;
                std::memcpy(ptr_dst_low, ptr_src_low, 256 * 256 * sizeof(float));

                //step3. 取high res mask, 进行地址偏移
                float* ptr_dst_high       = high_res_masks_data.data() + idx_batch * 1024 * 1024;
                const float* ptr_src_high = ptr_high_res_masks + idx_batch * 1024 * 1024;
                std::memcpy(ptr_dst_high, ptr_src_high, 1024 * 1024 * sizeof(float));

                //step4. 取sam token out, 进行地址偏移
                float* ptr_dst_samtoken   = sam_output_data.data() + idx_batch * 256;
                const float* ptr_src_samtoken = ptr_sam_token + idx_batch * 256;
                std::memcpy(ptr_dst_samtoken, ptr_src_samtoken, 256 * sizeof(float));

                //step5. 处理卡尔曼信息
                tracked_stracks[idx_batch].single_predict();
                std::vector<float> tlwh_;
                bool found_non_zero = found_tlwh(ptr_high_res_masks + idx_batch * 1024 * 1024, tlwh_);

                if(this->infer_status.stable_ious_threshold < ious_data[best_iou_ind])
                {
                    tracked_stracks[idx_batch].update(tlwh_, this->infer_status.current_frame);
                    tracked_stracks[idx_batch].stable_frames += 1;
                }
                else
                {
                    tracked_stracks[idx_batch].stable_frames = 0;
                }

                if(true != std::isnan(ious_data[best_iou_ind]))
                {
                    KfResultout.best_iou.push_back(ious_data[best_iou_ind]);
                }
                else
                {
                    KfResultout.best_iou.push_back(0);
                }
            }
            else
            {
                //step1.
                std::vector<float> ious_data;
                for (size_t idx_ious = 0; idx_ious < 3; ++idx_ious)
                {
                    ious_data.push_back(ptr_ious[idx_ious + idx_batch * 3]);
                }

                tracked_stracks[idx_batch].single_predict();
                std::vector<std::vector<float>> tlbrs_;

                for(size_t idx_high_muti_masks = 0; idx_high_muti_masks < 3; idx_high_muti_masks++)
                {
                    std::vector<float> tlbr;
                    bool found_non_zero = found_tlwh(high_res_multimasks_data + idx_batch * 3 * 1024 * 1024 + idx_high_muti_masks * 1024 * 1024, tlbr);
                    tlbrs_.push_back(tlbr);
                }

                std::vector<float> kf_ious;
                kf_ious = tracked_stracks[idx_batch].compute_iou(tlbrs_);
                std::vector<float> weighted_ious;
                for(size_t idx_kf_ious = 0; idx_kf_ious < kf_ious.size(); idx_kf_ious++)
                {
                    float weighted_iou = this->infer_status.kf_score_weight * kf_ious[idx_kf_ious] + 
                                        (1 - this->infer_status.kf_score_weight) * ious_data[idx_kf_ious];
                    weighted_ious.push_back(weighted_iou);
                }
                auto max_weighted_iou = std::max_element(weighted_ious.begin(), weighted_ious.end());
                size_t best_weighted_iou_ind = std::distance(weighted_ious.begin(), max_weighted_iou);

                //进行地址偏移    low_res_masks_data [this->batch_size, 1, 256, 256]    ptr_low_res_multimasks [this->batch_size, 3, 256, 256]
                float* ptr_dst_low = low_res_masks_data.data() + idx_batch * 256 * 256;
                const float* ptr_src_low = ptr_low_res_multimasks + (best_weighted_iou_ind + idx_batch * 3) * 256 * 256;

                std::memcpy(ptr_dst_low, ptr_src_low, 256 * 256 * sizeof(float));

                float* ptr_dst_high = high_res_masks_data.data() + idx_batch * 1024 * 1024;
                const float* ptr_src_high = high_res_multimasks_data + best_weighted_iou_ind * 1024 * 1024 + idx_batch * 3 * 1024 * 1024;
                std::memcpy(ptr_dst_high, ptr_src_high, 1024 * 1024 * sizeof(float));

                float* ptr_dst_samtoken   = sam_output_data.data() + idx_batch * 256;
                const float* ptr_src_samtoken = ptr_sam_tokens + idx_batch * 3 * 256 + best_weighted_iou_ind * 256;
                std::memcpy(ptr_dst_samtoken, ptr_src_samtoken, 256 * sizeof(float));

                if(true != std::isnan(ious_data[best_weighted_iou_ind]))
                {
                    auto max_iou = std::max_element(ious_data.begin(), ious_data.end());
                    KfResultout.best_iou.push_back(*max_iou);
                }
                else
                {
                    KfResultout.best_iou.push_back(0);
                }

                if(this->infer_status.stable_ious_threshold < ious_data[best_weighted_iou_ind])
                {
                    tracked_stracks[idx_batch].update(tlbrs_[best_weighted_iou_ind], this->infer_status.current_frame);
                    this->infer_status.stable_frames += 1;
                }
                else
                {
                    this->infer_status.stable_frames = 0;
                }
            }
        }

        // 创建 low_res_masks_tensor 的形状
        std::vector<int64_t> low_res_masks_shape = {this->batch_size, 1, 256, 256};
        Ort::Value low_res_masks_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            low_res_masks_data.data(),
            low_res_masks_data.size(),
            low_res_masks_shape.data(),  // 使用 std::vector 管理形状
            low_res_masks_shape.size()   // 维度大小
        );

        // 创建 high_res_masks_tensor 的形状
        std::vector<int64_t> high_res_masks_shape = {this->batch_size, 1, 1024, 1024};
        Ort::Value high_res_masks_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            high_res_masks_data.data(),
            high_res_masks_data.size(),
            high_res_masks_shape.data(),  // 使用 std::vector 管理形状
            high_res_masks_shape.size()   // 维度大小
        );


        float* ptr_dst_samtoken   = sam_output_data.data() + 0 * 256;
        const float* ptr_src_samtoken = ptr_sam_token + 0 * 256;
        std::memcpy(ptr_dst_samtoken, ptr_src_samtoken, 256 * sizeof(float));
        // 创建 sam_output_token_tensor 的形状
        std::vector<int64_t> sam_output_token_shape = {this->batch_size, 256};
        Ort::Value sam_output_token_tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            sam_output_data.data(),
            sam_output_data.size(),
            sam_output_token_shape.data(),  // 使用 std::vector 管理形状
            sam_output_token_shape.size()   // 维度大小
        );

        //KfResultout.best_iou = 1;    //todo 暂时禁用记忆帧优化 会影响多目标效果// KfResultout.best_iou / this->batch_size;    //这个会影响单目标的效果
        KfResultout.obj_score = ptr_object_score_logits[0];    //TODO yanwendou
        KfResultout.low_res_masks.push_back(std::move(low_res_masks_tensor));
        KfResultout.high_res_masks.push_back(std::move(high_res_masks_tensor));
        KfResultout.sam_output_token.push_back(std::move(sam_output_token_tensor));
        KfResultout.object_score_logits.push_back(std::move(decoderKfIn[0]));
    }

    //4. 确定out形式
    //return kftmp;
    /*
        "iou"
        "score"
        "low_res_masks", 
        "high_res_masks", 
        "sam_output_token"
    */
}
// input:
//      mask_for_mem    [1,1,1024,1024]
//      pix_feat        [1,256,64,64]

// output:
//      maskmem_features  [1,64,64,64]
//      maskmem_pos_enc   [4096,1,64]
//      temporal_code     [7,1,1,64]


cv::Rect findBoundingBox(cv::Mat& outimg) {
    // 1. 找到所有灰度值非零的像素
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(outimg, nonZeroPoints);

    // 2. 计算包含所有非零像素的最小包围盒
    if (!nonZeroPoints.empty()) {
        return cv::boundingRect(nonZeroPoints);
    }
    // 如果没有非零像素，返回空矩形
    return cv::Rect();
}

// 复用所有过程中的gpu内存
void Sam2Singleton::gpu_preprocess_sync(
    const cv::Mat& input,
    float* output_ptr  // 预分配内存地址：image_preprocess_data.data()
) {
    // 初始化页锁定内存
    if(page_locked_buffer.empty()) {
        page_locked_buffer.create(1, 3*1024*1024, CV_32F);
        cv::cuda::registerPageLocked(page_locked_buffer);
    }

    // 第1步：上传（同步）
    cv::cuda::GpuMat& d_input = gpu_buffers[0];
    if(d_input.empty() || d_input.size() != input.size() || d_input.type() != input.type()) {
        d_input.create(input.size(), input.type());
    }
    d_input.upload(input); // 移除stream参数

    // 第2步：调整尺寸（同步）
    cv::cuda::GpuMat& resized = gpu_buffers[1];
    const cv::Size target_size(1024, 1024);
    if(resized.empty() || resized.size() != target_size || resized.type() != input.type()) {
        resized.create(target_size, input.type());
    }
    cv::cuda::resize(d_input, resized, target_size); // 移除stream参数

    // 第3步：类型转换+颜色空间（同步）
    cv::cuda::GpuMat& converted = gpu_buffers[2];
    if(converted.empty() || converted.size() != target_size || converted.type() != CV_32FC3) {
        converted.create(target_size, CV_32FC3);
    }
    resized.convertTo(converted, CV_32F, 1.0/255.0); // 移除stream参数
    cv::cuda::cvtColor(converted, converted, cv::COLOR_BGR2RGB); // 移除stream参数

    // 第4步：通道分割（同步）
    for(auto& ch : channel_buffers) {
        if(ch.empty() || ch.size() != converted.size()) {
            ch.create(converted.size(), CV_32F);
        }
    }
    cv::cuda::split(converted, channel_buffers); // 移除stream参数

    // 第5步：平面格式合并（同步）
    cv::cuda::GpuMat& blob_gpu = gpu_buffers[3];
    if(blob_gpu.empty() || blob_gpu.cols != 3*1024*1024) {
        blob_gpu.create(1, 3*1024*1024, CV_32F);
    }
    
    // 同步拷贝通道数据
    channel_buffers[0].reshape(1,1).copyTo(blob_gpu.colRange(0, 1024*1024));
    channel_buffers[1].reshape(1,1).copyTo(blob_gpu.colRange(1024*1024, 2*1024*1024));
    channel_buffers[2].reshape(1,1).copyTo(blob_gpu.colRange(2*1024*1024, 3*1024*1024));

    // 第6步：下载（同步）
    cv::cuda::GpuMat& download_buf = gpu_buffers[4];
    if(download_buf.empty() || download_buf.size() != blob_gpu.size()) {
        download_buf.create(blob_gpu.size(), blob_gpu.type());
    }
    blob_gpu.copyTo(download_buf); // 同步拷贝
}

void Sam2Singleton::gpu_preprocess_async(
    const cv::Mat& input,
    float*       output_ptr  // 预分配内存地址：image_preprocess_data.data()
) {
    // —— 2. 准备一个单一 stream 和缓冲区引用 ——
    cv::cuda::GpuMat& d_input      = gpu_buffers[0];
    cv::cuda::GpuMat& resized      = gpu_buffers[1];
    cv::cuda::GpuMat& converted    = gpu_buffers[2];
    cv::cuda::GpuMat& blob_gpu     = gpu_buffers[3];
    cv::cuda::GpuMat& download_buf = gpu_buffers[4];
    const cv::Size   target_size(1024,1024);
    const int        planeSize = target_size.area();

    // —— 3. 确保所有 GPU Mats 大小／类型正确 ——
    if (d_input.empty()   || d_input.size()   != input.size()    || d_input.type()   != input.type())
        d_input.create(input.size(), input.type());
    if (resized.empty()   || resized.size()   != target_size       || resized.type()   != input.type())
        resized.create(target_size, input.type());
    if (converted.empty() || converted.size() != target_size       || converted.type() != CV_32FC3)
        converted.create(target_size, CV_32FC3);
    if (blob_gpu.empty()  || blob_gpu.cols    != 3 * planeSize)
        blob_gpu.create(1, 3 * planeSize, CV_32F);
    if (download_buf.empty() || download_buf.size() != blob_gpu.size() || download_buf.type() != blob_gpu.type())
        download_buf.create(blob_gpu.size(), blob_gpu.type());
    for (auto& ch : channel_buffers) {
        if (ch.empty() || ch.size() != target_size)
            ch.create(target_size, CV_32F);
    }

    // —— 4. 异步流水线：upload → resize → convert → cvtColor → split → merge ——
    d_input.upload(input, stream);
    cv::cuda::resize(d_input, resized, target_size, 0, 0, cv::INTER_LINEAR, stream);
    resized.convertTo(converted, CV_32F, 1.0f/255.0f, 0.0f, stream);
    cv::cuda::cvtColor(converted, converted, cv::COLOR_BGR2RGB, /*dstCn=*/0, stream);
    cv::cuda::split(converted, channel_buffers, stream);

    // CHW 平面合并到一行
    for (int c = 0; c < 3; ++c) {
        channel_buffers[c]
            .reshape(1,1)
            .copyTo(blob_gpu.colRange(c*planeSize, (c+1)*planeSize), stream);
    }

    // —— 5. 异步下载到 page‑locked host buffer ——
    blob_gpu.copyTo(download_buf, stream);

    // —— 6. 一次同步 等待所有操作完成 —— 
    stream.waitForCompletion();
}

void Sam2Singleton::preprocess(cv::Mat &image){
    // cv::Mat blob_cpu(1, 3 * 1024 * 1024, CV_32F, image_preprocess_data.data());

    // cv::dnn::blobFromImage(image, blob_cpu, 1.0 / 255.0, cv::Size(1024, 1024), cv::Scalar(0, 0, 0), true, false, CV_32F);

    gpu_preprocess_sync(image, image_preprocess_data.data());    //sync faster than async

    // this->input_images.clear();
    // this->input_images.emplace_back(blob_cpu);
}

void Sam2Singleton::postprocess(std::vector<Ort::Value> &output_tensors){
/*
    //源码后处理
    non_zero_indices = np.argwhere(mask)
    if len(non_zero_indices) == 0:
        bbox = [0, 0, 0, 0]
    else:
        y_min, x_min = non_zero_indices.min(axis=0).tolist()
        y_max, x_max = non_zero_indices.max(axis=0).tolist()
        bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
*/
    for(size_t i = 0; i < this->batch_size; i++)
    {
        float* output =  output_tensors[0].GetTensorMutableData<float>();    //可变数据
        cv::Mat outimg(this->ori_img->size(),CV_32FC1,output + i * this->ori_img->cols * this->ori_img->rows);

        /* gpu process*/
        if (!pinned_ptr) {
            if (pinned_ptr) cudaFreeHost(pinned_ptr);  // 释放旧内存
            cudaHostAlloc((void**)&pinned_ptr,  outimg.rows * outimg.cols, cudaHostAllocDefault);
            page_locked_bufferPostProcess = cv::Mat(outimg.rows, outimg.cols, CV_8UC1, pinned_ptr);
        }
        /* gpu process*/

        outimg.convertTo(page_locked_bufferPostProcess, CV_8UC1, 255);

        cv::Rect bbox = getOutBBoxFromSam2Out(page_locked_bufferPostProcess);

        #ifdef DRAW_PERSON
        cv::Scalar color = g_vcolors[i % g_vcolors.size()];
        cv::rectangle(*ori_img, bbox, color, 4);
        #endif

        this->LastRect = bbox;
        // //draw出 kf bbox
        // std::vector<float> pred_bbox = tracked_stracks[i].mean_to_xyxy();
        // cv::Rect bboxKF;
        // bboxKF.x = pred_bbox[0] * this->ori_img->cols / 1024.0;
        // bboxKF.y = pred_bbox[1] * this->ori_img->rows / 1024.0;
        // bboxKF.width = pred_bbox[2] * this->ori_img->cols / 1024.0 - bboxKF.x;
        // bboxKF.height = pred_bbox[3] * this->ori_img->rows / 1024.0 - bboxKF.y;
        // cv::rectangle(*ori_img, bboxKF, cv::Scalar(0,255,0),2);
    }

    // cv::threshold(dst,dst,0,255,cv::THRESH_BINARY);
    // cv::cvtColor(dst, dst1, cv::COLOR_GRAY2BGR);
    // dst1.copyTo(*ori_img);
}

/* gpu process*/
cv::Rect Sam2Singleton::getOutBBoxFromSam2Out(cv::Mat& outimg){
    // cv::Mat dst;
    // cv::threshold(outimg, dst, 200, 255,cv::THRESH_BINARY);
    // cv::Rect bbox = findBoundingBox(dst);
    // return bbox; // 空矩形

    // 上传图像到 GPU，只在尺寸或类型变更时重新创建
    if (d_outimg.empty() || d_outimg.size() != outimg.size() || d_outimg.type() != CV_8UC1) {
        d_outimg.create(outimg.size(), CV_8UC1);
    }
    d_outimg.upload(outimg);

    // 二值化图像
    if (d_bin.empty() || d_bin.size() != outimg.size()) {
        d_bin.create(outimg.size(), CV_8UC1);
    }
    cv::cuda::threshold(d_outimg, d_bin, 200, 255, cv::THRESH_BINARY);

    d_bin.download(outimg);
    // 计算非零区域
    std::vector<cv::Point> nonZeroPoints;
    cv::findNonZero(outimg, nonZeroPoints);  // 在 GPU 上查找非零点

    if (!nonZeroPoints.empty()) {
        return cv::boundingRect(nonZeroPoints);
    }

    return cv::Rect(); // 空矩形
}

int Sam2Singleton::setparms(std::vector<ParamsSam2>& parms){
    this->parms = parms;
    this->batch_size = this->parms.size();
    setparms_AllocateBatch_One();
    return 1;
}

bool found_tlwh(const float *ptr,std::vector<float>& tlwh_)
{
    //获取high_res_masks_data的bbox
    int y_min = std::numeric_limits<int>::max();
    int x_min = std::numeric_limits<int>::max();
    int y_max = std::numeric_limits<int>::min();
    int x_max = std::numeric_limits<int>::min();

    bool found_non_zero = false;

    int height = 1024;
    int width  = 1024;

    for (int y = 0; y < height; ++y) {     //TODO yanwendou 这个图像的访问x、y对不对
        for (int x = 0; x < width; ++x) {
            int index = y * width + x;
            if (ptr[index] > 0.0f) {
                // 如果找到非零元素，更新边界框
                if (y < y_min) y_min = y;
                if (x < x_min) x_min = x;
                if (y > y_max) y_max = y;
                if (x > x_max) x_max = x;
                found_non_zero = true;
            }
        }
    }

    if (!found_non_zero) {
        y_min = x_min = y_max = x_max = 0;
    }

    tlwh_ = {(float)x_min, (float)y_min, (float)(x_max - x_min), (float)(y_max - y_min)};
    
    return found_non_zero;
}

void print_tensor_shape(Ort::Value& tensor_value) {
    // 获取 tensor 的 shape
    const Ort::TensorTypeAndShapeInfo& tensor_info = tensor_value.GetTensorTypeAndShapeInfo();
    
    // 获取 tensor 的 rank（维度数量）
    size_t num_dims = tensor_info.GetDimensionsCount();
    std::cout << "Tensor rank (number of dimensions): " << num_dims << std::endl;
    
    // 获取每个维度的大小
    std::vector<int64_t> shape(num_dims);
    tensor_info.GetDimensions(shape.data(), num_dims);
    
    // 打印每个维度的大小
    std::cout << "Tensor shape: ";
    for (size_t i = 0; i < num_dims; ++i) {
        std::cout << shape[i] << " ";
    }
    std::cout << std::endl;
}


// 构造函数：加载引擎文件并初始化
TensorRTInference::TensorRTInference(const std::string& engineFilePath, int gpuId) {
    this->gpuId = gpuId;
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);

    // 加载引擎文件
    std::vector<char> engineData = loadEngine(engineFilePath);

    // 创建 TensorRT 运行时
    runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(logger));
    if (!runtime) {
        throw std::runtime_error("Failed to create TensorRT runtime");
    }

    // 反序列化引擎
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    if (!engine) {
        throw std::runtime_error("Failed to deserialize CUDA engine");
    }

    // 创建执行上下文
    context = std::unique_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        throw std::runtime_error("Failed to create execution context");
    }

    //TOOYWD 提前分配的话 容易内存不足
    // // 分配输入输出缓冲区
    // int numBindings = engine->getNbIOTensors();
    // buffers.resize(numBindings);
    // for (int i = 0; i < numBindings; ++i) {
    //     const char* tensorName = engine->getIOTensorName(i);
    //     nvinfer1::Dims dims = engine->getTensorShape(tensorName);
    //     size_t size = 1;
    //     for (int j = 0; j < dims.nbDims; ++j) {
    //         size *= dims.d[j];
    //     }
    //     size *= sizeof(float); // 假设数据类型是 float

    //     buffers[i] = allocateDeviceMemory(size);
    // }
}

// 加载引擎文件
std::vector<char> TensorRTInference::loadEngine(const std::string& engineFilePath) {
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);

    std::ifstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        throw std::runtime_error("Failed to open engine file: " + engineFilePath);
    }

    engineFile.seekg(0, std::ios::end);
    size_t fileSize = engineFile.tellg();
    engineFile.seekg(0, std::ios::beg);

    std::vector<char> engineData(fileSize);
    engineFile.read(engineData.data(), fileSize);
    engineFile.close();

    return engineData;
}

// 分配 GPU 内存
void* TensorRTInference::allocateDeviceMemory(size_t size) {
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    void* deviceMemory;
    cudaMalloc(&deviceMemory, size);
    return deviceMemory;
}

// 释放 GPU 内存
void TensorRTInference::freeDeviceMemory(void* deviceMemory) {
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    cudaFree(deviceMemory);
}

// 推理函数：输入图片，返回推理结果
bool TensorRTInference::MemAttentioninfer(std::vector<Ort::Value>& input_tensor, float* ptr_tensorrt_mem_attention_data) {
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    #ifdef TIME_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // 输入张量名称列表
    std::vector<std::string> inputNames = {
        "current_vision_feat",
        "current_vision_pos_embed",
        "memory_0",
        "memory_1",
        "memory_pos_embed"
    };

    // 输出张量名称
    const char* outputTensorName = "image_embed"; // 替换为你的输出张量名称

    // 设置输入张量的形状
    std::vector<nvinfer1::Dims> inputShapes = {
        {4, {this->batch_size, 256, 64, 64}},         // current_vision_feat
        {3, {4096, this->batch_size, 256}},          // current_vision_pos_embed
        {3, {obj_size, this->batch_size, 256}},             // memory_0
        {5, {mem_feature_size, this->batch_size, 64, 64, 64}},      // memory_1
        {3, {obj_size * 4 + mem_feature_size * 4096, this->batch_size, 64}},           // memory_pos_embed
    };

    // 处理输入张量
    #ifdef TIME_DEBUG
    auto start222 = std::chrono::high_resolution_clock::now();
    #endif

    #ifdef TENSORRT_8_X
    std::vector<void *> device_buffers_MemAttention;
    #endif

    for (size_t i = 0; i < inputNames.size(); ++i) {
        // 设置输入张量的形状
        #ifdef TENSORRT_8_X
        context->setBindingDimensions(i, inputShapes[i]);
        #else
        context->setInputShape(inputNames[i].c_str(), inputShapes[i]);
        #endif

        // 使用已分配的内存
        void* gpuMemory = inputData[i];
        size_t inputSize = inputSizes[i];

        // 设置输入张量的地址
        #ifdef TENSORRT_8_X
        device_buffers_MemAttention.push_back(gpuMemory);
        #else
        context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
        #endif

        // 将输入数据从主机拷贝到 GPU
        cudaMemcpy(gpuMemory, input_tensor[i].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
    }

    #ifdef TIME_DEBUG
    #ifdef D2HORH2D
    auto end222 = std::chrono::high_resolution_clock::now();
    auto duration222 = std::chrono::duration_cast<std::chrono::milliseconds>(end222 - start222);
    std::cout << "Time taken cudaMemcpy host to gpu: "  << duration222.count() << " milliseconds." << std::endl;
    #endif
    #endif

    // 处理输出张量
    void* gpuMemory = outputData[0]; // 假设只有一个输出张量
    size_t outputSize = outputSizes[0];

    // 设置输出张量的地址

    #ifdef TENSORRT_8_X
    device_buffers_MemAttention.push_back(gpuMemory);
    #else
    context->setTensorAddress(outputTensorName, gpuMemory);
    #endif
    // 执行推理

    #ifdef TENSORRT_8_X
    if (!context->enqueueV2(device_buffers_MemAttention.data(), stream, nullptr))
    {
        throw std::runtime_error("Failed to enqueue inference with enqueueV2");
    }
    #else    //10.x
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to enqueue inference");
    }
    #endif

    #ifdef TIME_DEBUG
    auto start111 = std::chrono::high_resolution_clock::now();
    #endif

    // // 将输出数据从 GPU 拷贝到 CPU
    cudaMemcpy(ptr_tensorrt_mem_attention_data, gpuMemory, outputSize, cudaMemcpyDeviceToHost);
    #ifdef TIME_DEBUG
    #ifdef D2HORH2D
    auto end111 = std::chrono::high_resolution_clock::now();
    auto duration111 = std::chrono::duration_cast<std::chrono::milliseconds>(end111 - start111);
    std::cout << "Time taken cudaMemcpy to host memory_attention: " << duration111.count() << " milliseconds." << std::endl;
    #endif
    #endif

    // 打印推理时间
    #ifdef TIME_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "TensorRT inference time: " << duration.count() << " ms" << std::endl;
    #endif

    return true;
}

void TensorRTInference::infer_ImageEncoder(void* gpuPtr_image, std::vector<int>& batch_size_info, std::vector<std::vector<float>>& outputHostData, TRACKTYPEBYSAM2 trackType) {
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    #ifdef TIME_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // 输入张量名称列表
    std::vector<std::string> inputNames = {"image", "batch_size"};
    // 设置输入张量的形状
    std::vector<nvinfer1::Dims> inputShapes = {
        {4, {1, 3, 1024, 1024}},  // image: [batch_size, channels, height, width]
        {1, {(int64_t)batch_size_info.size()}}  // batch_size: [batch_size] -- [N]
    };


    // 输出张量名称列表
    std::vector<std::string> outputNames = {
        "pix_feat",
        "high_res_feat0",
        "high_res_feat1",
        "vision_feats",
        "vision_pos_embed"
    };

    #ifdef TENSORRT_8_X
    std::vector<void *>device_buffers_ImageEncoder;
    #endif

    // 处理输入张量
    for (size_t i = 0; i < inputNames.size(); ++i) {

        if (inputNames[i] == "batch_size" && trackType == TRACKTYPEBYSAM2::SingleTrack) {
            continue;
        }

        // 设置输入张量的形状
        #ifdef TENSORRT_8_X
        context->setBindingDimensions(i, inputShapes[i]);
        #else
        context->setInputShape(inputNames[i].c_str(), inputShapes[i]);
        #endif

        // 使用已分配的内存
        void* gpuMemory = inputData[i];
        size_t inputSize = inputSizes[i];

        if (inputNames[i] == "image") {
            #ifdef TENSORRT_8_X
            device_buffers_ImageEncoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuPtr_image);
            #endif
        } else if (inputNames[i] == "batch_size") {
            cudaMemcpy(gpuMemory, batch_size_info.data(), inputSize, cudaMemcpyHostToDevice);
            #ifdef TENSORRT_8_X
            device_buffers_ImageEncoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
            #endif
        }
    }

    // 处理输出张量
    for (size_t i = 0; i < outputNames.size(); ++i) {
        // 使用已分配的内存
        void* gpuMemory = outputData[i];
        size_t outputSize = outputSizes[i];

        // 设置输出张量的地址
        #ifdef TENSORRT_8_X
        device_buffers_ImageEncoder.push_back(gpuMemory);
        #else
        context->setTensorAddress(outputNames[i].c_str(), gpuMemory);
        #endif
    }

    // 执行推理
    #ifdef TENSORRT_8_X
    if (!context->enqueueV2(device_buffers_ImageEncoder.data(), stream, nullptr)) 
    {
        throw std::runtime_error("Failed to enqueue inference with enqueueV2");
    }
    #else    //10.x
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to enqueue inference");
    }
    #endif

    // 将输出数据从 GPU 拷贝到 CPU
    for (size_t i = 0; i < outputNames.size(); ++i) {

        // 使用已分配的大小
        size_t outputSize = outputSizes[i];

        // 将输出数据从 GPU 拷贝到 CPU
        cudaMemcpy(outputHostData[i].data(), outputData[i], outputSize, cudaMemcpyDeviceToHost);
    }


    // 打印推理时间
    #ifdef TIME_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "TensorRT ImageEncoder inference time: " << duration.count() << " ms" << std::endl;
    #endif

    return;
}

void TensorRTInference::infer_ImageDecoderFirst(std::vector<float>& point_val,
                                                std::vector<int>& point_labels,
                                                std::vector<Ort::Value>& mem_attention_out,
                                                std::vector<std::vector<float>>& outputDataForImageDecoder_First) 
{
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    #ifdef TIME_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // 输入张量名称列表
    std::vector<std::string> inputNames = {
        "point_coords",
        "point_labels",
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1"
    };

    // 输出张量名称列表
    std::vector<std::string> outputNames = {
        "object_score_logits",
        "low_res_multimasks",
        "high_res_multimasks", 
        "sam_output_tokens",
        "ious", 
        "low_res_masks", 
        "high_res_masks", 
        "sam_output_token"
    };

    // 设置输入张量的形状
    std::vector<nvinfer1::Dims> inputShapes = {
        {3, {this->batch_size, 2, 2}},  
        {2, {this->batch_size, 2}},
        {4, {this->batch_size, 256, 64, 64}}, 
        {4, {this->batch_size, 32, 256, 256}}, 
        {4, {this->batch_size, 64, 128, 128}}
    };

    // 处理输入张量
    for (size_t i = 0; i < inputNames.size(); ++i) {
        // 设置输入张量的形状
        #ifdef TENSORRT_8_X
        context->setBindingDimensions(i, inputShapes[i]);
        #else
        context->setInputShape(inputNames[i].c_str(), inputShapes[i]);
        #endif

        // 使用已分配的内存
        void* gpuMemory = inputData[i];
        size_t inputSize = inputSizes[i];

        // 设置输入张量的地址
        // context->setTensorAddress(inputNames[i].c_str(), gpuMemory);

        // 将输入数据从主机拷贝到 GPU
        if (inputNames[i] == "point_coords") {
            cudaMemcpy(gpuMemory, point_val.data(), inputSize, cudaMemcpyHostToDevice);
        } else if (inputNames[i] == "point_labels") {
            cudaMemcpy(gpuMemory, point_labels.data(), inputSize, cudaMemcpyHostToDevice);
        } else if (inputNames[i] == "image_embed") {
            cudaMemcpy(gpuMemory, mem_attention_out[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        } else if (inputNames[i] == "high_res_feats_0") {
            cudaMemcpy(gpuMemory, mem_attention_out[1].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        } else if (inputNames[i] == "high_res_feats_1") {
            cudaMemcpy(gpuMemory, mem_attention_out[2].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        }
    }

    // 处理输出张量
    for (size_t i = 0; i < outputNames.size(); ++i) {
        // 使用已分配的内存
        void* gpuMemory = outputData[i];
        size_t outputSize = outputSizes[i];

        // 设置输出张量的地址
        // context->setTensorAddress(outputNames[i].c_str(), gpuMemory);
    }

    // 执行推理
    // if (!context->enqueueV3(stream)) {
    //     throw std::runtime_error("Failed to enqueue inference");
    // }

    // 将输出数据从 GPU 拷贝到 CPU
    for (size_t i = 0; i < outputNames.size(); ++i) {
        // 使用已分配的大小
        size_t outputSize = outputSizes[i];

        // 将输出数据从 GPU 拷贝到 CPU
        cudaMemcpy(outputDataForImageDecoder_First[i].data(), outputData[i], outputSize, cudaMemcpyDeviceToHost);
    }


    // 打印推理时间
    #ifdef TIME_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "TensorRT infer_ImageDecoderFirst time: " << duration.count() << " ms" << std::endl;
    #endif

    return;
}

void Sam2Singleton::creatImageDecoder_First(std::vector<Ort::Value> &ImageDecoder_First_out)
{
    std::vector<nvinfer1::Dims> outputShapes = {
        {2, {this->batch_size, 1}},                     // object_score_logits: [3, 1]
        {4, {this->batch_size, 3, 256, 256}},           // low_res_multimasks: [3, 3, 256, 256]
        {4, {this->batch_size, 3, 1024, 1024}},         // high_res_multimasks: [3, 3, 1024, 1024]
        {3, {this->batch_size, 3, 256}},                // sam_output_tokens: [3, 3, 256]
        {2, {this->batch_size, 3}},                     // ious: [3, 3]
        {4, {this->batch_size, 1, 256, 256}},           // low_res_masks: [3, 1, 256, 256]
        {4, {this->batch_size, 1, 1024, 1024}},         // high_res_masks: [3, 1, 1024, 1024]
        {2, {this->batch_size, 256}}                    // sam_output_token: [3, 256]
    };

    // 为每个输出张量创建 Ort::Value
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        // 获取当前输出的形状和数据
        auto& shape = outputShapes[i];
        auto& data = outputDataForImageDecoder_First[i];

        #ifdef TENSORRT_8_X
        // 将 int32_t 数组 shape.d 转换为 int64_t 类型的 vector
        std::vector<int64_t> shape_d_int64;
        for (int j = 0; j < shape.nbDims; ++j)
        {
            shape_d_int64.push_back(static_cast<int64_t>(shape.d[j]));
        }

        // 创建 Ort::Value
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,                   // 内存信息
            data.data(),                   // 数据指针
            data.size(),                   // 数据大小
            shape_d_int64.data(),          // 转换后的维度信息
            shape_d_int64.size()           // 维度数量
        );
        #else
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            data.data(),
            data.size(),
            shape.d,
            shape.nbDims);
        #endif
        // 将 Ort::Value 添加到输出向量
        ImageDecoder_First_out.push_back(std::move(tensor));
    }
}

void TensorRTInference::infer_ImageDecoder(std::vector<float>& point_val,
    std::vector<int>&   point_labels,
    std::vector<Ort::Value>& mem_attention_out,
    std::vector<std::vector<float>>& outputDataForImageDecoder_Second, 
    std::vector<void *>&inputStr)
{
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    #ifdef TIME_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // 输入张量名称列表
    std::vector<std::string> inputNames = {
        "point_coords",
        "point_labels",
        "image_embed",
        "high_res_feats_0",
        "high_res_feats_1"
    };

    // 输出张量名称列表
    std::vector<std::string> outputNames = {
        "obj_ptr", 
        "mask_for_mem", 
        "pred_mask",
        "ious", 
    };

    // 设置输入张量的形状
    std::vector<nvinfer1::Dims> inputShapes = {
        {3, {this->batch_size, 2, 2}},  
        {2, {this->batch_size, 2}},
        {4, {this->batch_size, 256, 64, 64}}, 
        {4, {this->batch_size, 32, 256, 256}}, 
        {4, {this->batch_size, 64, 128, 128}}
    };

    std::vector<int> frame_size = {1080, 1920}; 

    #ifdef TENSORRT_8_X
    std::vector<void *> device_buffers_ImageDecoder;
    #endif

    // 处理输入张量
    for (size_t i = 0; i < inputNames.size(); ++i) {
        // 设置输入张量的形状
        #ifdef TENSORRT_8_X
        context->setBindingDimensions(i, inputShapes[i]);
        #else
        context->setInputShape(inputNames[i].c_str(), inputShapes[i]);
        #endif

        // 使用已分配的内存
        void* gpuMemory = inputData[i];
        size_t inputSize = inputSizes[i];

        // 设置输入张量的地址

        // 将输入数据从主机拷贝到 GPU
        if (inputNames[i] == "point_coords") {
            cudaMemcpy(gpuMemory, point_val.data(), inputSize, cudaMemcpyHostToDevice);
            
            #ifdef TENSORRT_8_X
            device_buffers_ImageDecoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
            #endif
        } else if (inputNames[i] == "point_labels") {
            cudaMemcpy(gpuMemory, point_labels.data(), inputSize, cudaMemcpyHostToDevice);

            #ifdef TENSORRT_8_X
            device_buffers_ImageDecoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
            #endif
        } else if (inputNames[i] == "frame_size") {
            cudaMemcpy(inputData[i], frame_size.data(), inputSize, cudaMemcpyHostToDevice);
            #ifdef TENSORRT_8_X
            device_buffers_ImageDecoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
            #endif
        } else if (inputNames[i] == "image_embed") {
            // cudaMemcpy(gpuMemory, mem_attention_out[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
            // cudaMemcpy(gpuMemory, (float *)inputStr[0], inputSize, cudaMemcpyHostToDevice);

            #ifdef TENSORRT_8_X
            device_buffers_ImageDecoder.push_back(inputStr[0]);
            #else
            context->setTensorAddress(inputNames[i].c_str(), (float *)inputStr[0]);
            #endif
        } else if (inputNames[i] == "high_res_feats_0") {
            // cudaMemcpy(gpuMemory, mem_attention_out[1].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
            // cudaMemcpy(gpuMemory, (float *)inputStr[1], inputSize, cudaMemcpyHostToDevice);context->setTensorAddress(inputNames[i].c_str(), (float *)inputStr[0]);
            #ifdef TENSORRT_8_X
            device_buffers_ImageDecoder.push_back(inputStr[1]);
            #else
            context->setTensorAddress(inputNames[i].c_str(), (float *)inputStr[1]);
            #endif
        } else if (inputNames[i] == "high_res_feats_1") {
            // cudaMemcpy(gpuMemory, mem_attention_out[2].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
            // cudaMemcpy(gpuMemory, (float *)inputStr[2], inputSize, cudaMemcpyHostToDevice);
            #ifdef TENSORRT_8_X
            device_buffers_ImageDecoder.push_back(inputStr[2]);
            #else
            context->setTensorAddress(inputNames[i].c_str(), (float *)inputStr[2]);
            #endif
        }
    }
    //设置输出张量的形状
    // # obj_ptr              [3, 256]
    // # mask_for_mem         [3, 1, 1024, 1024]
    // # pred_mask            [3, 1, 1024, 1024]

    std::vector<nvinfer1::Dims> outputShapes = {
        {2, {this->batch_size, 256}},
        {4, {this->batch_size, 1, 1024, 1024}},
        {4, {this->batch_size, 1, 1080, 1920}},
        {2, {this->batch_size, 3}},
    };

    // 处理输出张量
    for (size_t i = 0; i < outputNames.size(); ++i) {
        void* gpuMemory = outputData[i];
        size_t outputSize = outputSizes[i];
        if (!outputData[i]) {
            throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
        }

        // 设置输出张量的地址
        #ifdef TENSORRT_8_X
        device_buffers_ImageDecoder.push_back(gpuMemory);
        #else
        context->setTensorAddress(outputNames[i].c_str(), gpuMemory);
        #endif
    }

    // 执行推理
    #ifdef TENSORRT_8_X
    if (!context->enqueueV2(device_buffers_ImageDecoder.data(), stream, nullptr)) 
    {
        throw std::runtime_error("Failed to enqueue inference with enqueueV2");
    }
    #else    //10.x
    if (!context->enqueueV3(stream)) {
        throw std::runtime_error("Failed to enqueue inference");
    }
    #endif
    // 将输出数据从 GPU 拷贝到 CPU
    for (size_t i = 0; i < outputNames.size(); ++i) {
        size_t outputSize = outputSizes[i];
        // 将输出数据从 GPU 拷贝到 CPU
        cudaMemcpy(outputDataForImageDecoder_Second[i].data(), outputData[i], outputSize, cudaMemcpyDeviceToHost);
    }

    // 打印推理时间
    #ifdef TIME_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "TensorRT infer_ImageDecoder time: " << duration.count() << " ms" << std::endl;
    #endif

    // 创建 Ort::Value 的代码
    return;
}


void Sam2Singleton::creatPointInput(std::vector<float> &point_val, std::vector<int> &point_labels)
{
    if (0 == this->infer_status.current_frame)
    {
        if(parms.size()){
            for(size_t i = 0; i < parms.size(); i++)
            {
                auto box = parms[i].prompt_box;

                box.x = 1024*((float)box.x / ori_img->cols);
                box.y = 1024*((float)box.y / ori_img->rows);
                box.width = 1024*((float)box.width / ori_img->cols);
                box.height = 1024*((float)box.height / ori_img->rows);
                point_val.push_back((float)box.x);
                point_val.push_back((float)box.y);
                point_val.push_back((float)box.x+box.width);
                point_val.push_back((float)box.y+box.height);

                point_labels.push_back(2);
                point_labels.push_back(3);
            }
        }
        else{
            throw std::runtime_error("prompt parms is empty, cannot process");
        }
    }
    
    //warning, 这里在非condition帧的时候 必须送入-1 不能用下面的needpoint来操作-1
    else
    {
        for(size_t i = 0; i < parms.size(); i++)
        {
            auto box = parms[i].prompt_box;

            box.x = 0;
            box.y = 0;
            box.width = 0;
            box.height = 0;
            point_val.push_back((float)box.x);
            point_val.push_back((float)box.y);
            point_val.push_back((float)box.x+box.width);
            point_val.push_back((float)box.y+box.height);

            point_labels.push_back(-1);
            point_labels.push_back(-1);
        }
    }
}

void TensorRTInference::infer_ImageDecoderSecond(sKfOut& result_kf, std::vector<std::vector<float>>& outputDataForImageDecoder_Second) {
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    #ifdef TIME_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif

    // 输入张量名称列表
    std::vector<std::string> inputNames = {
        "object_score_logits", 
        "frame_size", 
        "low_res_masks", 
        "high_res_masks", 
        "sam_output_token"
    };

    // 输出张量名称列表
    std::vector<std::string> outputNames = {
        "obj_ptr", 
        "mask_for_mem", 
        "pred_mask"
    };

    // 设置输入张量的形状
    std::vector<nvinfer1::Dims> inputShapes = {
        {2, {this->batch_size, 1}},  
        {1, {2}},
        {4, {this->batch_size, 1, 256, 256}}, 
        {4, {this->batch_size, 1, 1024, 1024}}, 
        {2, {this->batch_size, 256}}
    };

    std::vector<int> frame_size = {1080, 1920}; 
    // 处理输入张量
    for (size_t i = 0; i < inputNames.size(); ++i) {
        #ifdef TENSORRT_8_X
        context->setBindingDimensions(i, inputShapes[i]);
        #else
        context->setInputShape(inputNames[i].c_str(), inputShapes[i]);
        #endif

        void* gpuMemory = inputData[i];
        size_t inputSize = inputSizes[i];
        if (!inputData[i]) {
            throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
        }

        // 设置输入张量的地址
        // context->setTensorAddress(inputNames[i].c_str(), inputData[i]);

        // 将输入数据从主机拷贝到 GPU
        if (inputNames[i] == "object_score_logits") {
            // 将 OpenCV Mat 转换为 float 数组并拷贝到 GPU
            cudaMemcpy(inputData[i], result_kf.object_score_logits[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        } else if (inputNames[i] == "frame_size") {
            // 将 batch_size_info 拷贝到 GPU
            cudaMemcpy(inputData[i], frame_size.data(), inputSize, cudaMemcpyHostToDevice);
        }
        else if (inputNames[i] == "low_res_masks") {
                    // 将 batch_size_info 拷贝到 GPU
                    cudaMemcpy(inputData[i], result_kf.low_res_masks[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        }
        else if (inputNames[i] == "high_res_masks") {
                    // 将 batch_size_info 拷贝到 GPU
                    cudaMemcpy(inputData[i], result_kf.high_res_masks[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        }
        else if (inputNames[i] == "sam_output_token") {
                    // 将 batch_size_info 拷贝到 GPU
                    cudaMemcpy(inputData[i], result_kf.sam_output_token[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);
        }
    }

    //设置输出张量的形状
    // # obj_ptr              [3, 256]
    // # mask_for_mem         [3, 1, 1024, 1024]
    // # pred_mask            [3, 1, 1024, 1024]

    std::vector<nvinfer1::Dims> outputShapes = {
        {2, {this->batch_size, 256}},
        {4, {this->batch_size, 1, 1024, 1024}},
        {4, {this->batch_size, 1, 1080, 1920}},
    };

    // 处理输出张量
    for (size_t i = 0; i < outputNames.size(); ++i) {
        void* gpuMemory = outputData[i];
        size_t outputSize = outputSizes[i];
        if (!outputData[i]) {
            throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
        }

        // 设置输出张量的地址
        // context->setTensorAddress(outputNames[i].c_str(), gpuMemory);
    }

    // 执行推理
    // if (!context->enqueueV3(stream)) {
    //     throw std::runtime_error("Failed to enqueue inference");
    // }

    // 将输出数据从 GPU 拷贝到 CPU
    for (size_t i = 0; i < outputNames.size(); ++i) {
        size_t outputSize = outputSizes[i];
        // 将输出数据从 GPU 拷贝到 CPU
        cudaMemcpy(outputDataForImageDecoder_Second[i].data(), outputData[i], outputSize, cudaMemcpyDeviceToHost);
    }


    // 打印推理时间
    #ifdef TIME_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "TensorRT infer_ImageDecoderSecond time: " << duration.count() << " ms" << std::endl;
    #endif

    // 创建 Ort::Value 的代码
    return;
}

void Sam2Singleton::creatTensorImageDecoder_Second(std::vector<Ort::Value> &img_decoder_out)
{
    std::vector<nvinfer1::Dims> outputShapes = {
        {2, {this->batch_size, 256}},                // 示例1
        {4, {this->batch_size, 1, 1024, 1024}},     // 示例2
        {4, {this->batch_size, 1, 1080, 1920}},     // 示例3
    };

    // 为每个输出张量创建 Ort::Value
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        // 获取当前输出的形状和数据
        auto& shape = outputShapes[i];
        auto& data = outputDataForImageDecoder_Second[i];

        #ifdef TENSORRT_8_X
        // 将 int32_t 数组 shape.d 转换为 int64_t 类型的 vector
        std::vector<int64_t> shape_d_int64;
        for (int j = 0; j < shape.nbDims; ++j) {
            shape_d_int64.push_back(static_cast<int64_t>(shape.d[j]));
        }

        // 创建 Ort::Value
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,                   // 内存信息
            data.data(),                   // 数据指针
            data.size(),                   // 数据大小
            shape_d_int64.data(),          // 转换后的维度信息
            shape_d_int64.size()           // 维度数量
        );

        #else
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            data.data(),
            data.size(),
            shape.d,
            shape.nbDims
        );
        #endif

        // 将 Ort::Value 添加到输出向量
        img_decoder_out.push_back(std::move(tensor));
    }
}

void TensorRTInference::infer_MemoryDecoder(std::vector<Ort::Value>& MemoryDecoderIn, std::vector<std::vector<float>>& outputDataForMmeEncoder, std::vector<void *>&mem_encoder_inStr) {
    #ifdef TIME_DEBUG
    auto start = std::chrono::high_resolution_clock::now();
    #endif
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    // 输入张量名称列表
    std::vector<std::string> inputNames = {
        "mask_for_mem",
        "pix_feat"
    };

    // 输出张量名称列表
    std::vector<std::string> outputNames = {
        "maskmem_features",
        "maskmem_pos_enc",
        "temporal_code"
    };


    // 设置输入张量的形状
    std::vector<nvinfer1::Dims> inputShapes = {
        {4, {this->batch_size, 1, 1024, 1024}},  
        {4, {this->batch_size, 256, 64, 64}},
    };

    #ifdef TENSORRT_8_X
    std::vector<void *> device_buffers_MemEncoder;
    #endif
    // 处理输入张量
    for (size_t i = 0; i < inputNames.size(); ++i) {

        #ifdef TENSORRT_8_X
        context->setBindingDimensions(i, inputShapes[i]);
        #else
        context->setInputShape(inputNames[i].c_str(), inputShapes[i]);
        #endif

        // 使用已分配的内存
        void* gpuMemory = inputData[i];
        size_t inputSize = inputSizes[i];
    
        if (!inputData[i]) 
        {
            throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
        }

        // 将输入数据从主机拷贝到 GPU
        if (inputNames[i] == "mask_for_mem") {
            cudaMemcpy(gpuMemory, MemoryDecoderIn[0].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);

            #ifdef TENSORRT_8_X
            device_buffers_MemEncoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
            #endif
            // context->setTensorAddress(inputNames[i].c_str(), mem_encoder_inStr[0]);
        } else if (inputNames[i] == "pix_feat") {
            // context->setTensorAddress(inputNames[i].c_str(), mem_encoder_inStr[1]);
            cudaMemcpy(gpuMemory, MemoryDecoderIn[1].GetTensorData<float>(), inputSize, cudaMemcpyHostToDevice);

            #ifdef TENSORRT_8_X
            device_buffers_MemEncoder.push_back(gpuMemory);
            #else
            context->setTensorAddress(inputNames[i].c_str(), gpuMemory);
            #endif
        }
    }

    //设置输出张量的形状
    // # maskmem_features              [3, 1]
    // # maskmem_pos_enc               [3, 3, 256, 256]
    // # temporal_code                 [3, 3, 1024, 1024]

    // 处理输出张量
    for (size_t i = 0; i < outputNames.size(); ++i) {
        if (!outputData[i]) 
        {
            throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
        }

        #ifdef TENSORRT_8_X
        device_buffers_MemEncoder.push_back(outputData[i]);
        #else
        context->setTensorAddress(outputNames[i].c_str(), outputData[i]);
        #endif
    }

    #ifdef TENSORRT_8_X
    if (!context->enqueueV2(device_buffers_MemEncoder.data(), stream, nullptr)) 
    {
        throw std::runtime_error("Failed to enqueue inference with enqueueV2");
    }
    #else    //10.x
    if (!context->enqueueV3(stream))
    {
        throw std::runtime_error("Failed to enqueue inference with enqueueV3");
    }
    #endif

    // 将输出数据从 GPU 拷贝到 CPU
    for (size_t i = 0; i < outputNames.size(); ++i) {
        // 使用已分配的内存
        void* gpuMemory = outputData[i];
        size_t outputSize = outputSizes[i];

        // 将输出数据从 GPU 拷贝到 CPU
        cudaMemcpy(outputDataForMmeEncoder[i].data(), outputData[i], outputSize, cudaMemcpyDeviceToHost);
    }

    // 打印推理时间
    #ifdef TIME_DEBUG
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "TensorRT infer_MemoryEncoder time: " << duration.count() << " ms" << std::endl;
    #endif

    // 创建 Ort::Value 的代码
    return;
}

void Sam2Singleton::creatTensorMemEncoder(std::vector<Ort::Value> &MemEncoder_out)
{
    std::vector<nvinfer1::Dims> outputShapes = {
        {4, {this->batch_size, 64, 64, 64}},                  // 示例1
        {3, {4096, this->batch_size, 64}},                    // 示例2
        {4, {7, 1, 1, 64}},                                  // 示例3
    };
    
    // 为每个输出张量创建 Ort::Value
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        // 获取当前输出的形状和数据
        auto& shape = outputShapes[i];
        auto& data = outputDataForMmeEncoder[i];
    
        #ifdef TENSORRT_8_X
        // 将 int32_t 数组 shape.d 转换为 int64_t 类型的 vector
        std::vector<int64_t> shape_d_int64;
        for (int j = 0; j < shape.nbDims; ++j) {
            shape_d_int64.push_back(static_cast<int64_t>(shape.d[j]));
        }

        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,                 // 内存信息
            data.data(),                 // 数据指针
            data.size(),                 // 数据大小
            shape_d_int64.data(),        // 转换后的维度信息
            shape_d_int64.size()         // 维度数量
        );

        #else
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            data.data(),
            data.size(),
            shape.d,
            shape.nbDims
        );
        #endif


    
        // 将 Ort::Value 添加到输出向量
        MemEncoder_out.push_back(std::move(tensor));
    }
}

void Sam2Singleton::createTensorImgEncoder(std::vector<Ort::Value>& img_encoder_out) {
    // 定义输出张量的形状
    std::vector<std::vector<int64_t>> outputShapes = {
        {this->batch_size, 256, 64, 64},       // pix_feat
        {this->batch_size, 32, 256, 256},      // high_res_feat0
        {this->batch_size, 64, 128, 128},      // high_res_feat1
        {this->batch_size, 256, 64, 64},       // vision_feats
        {4096, this->batch_size, 256}          // vision_pos_embed
    };

    // 为每个输出张量创建 Ort::Value
    for (size_t i = 0; i < outputShapes.size(); ++i) {
        // 获取当前输出的形状和数据
        auto& shape = outputShapes[i];
        auto& data = outputHostData[i];

        // 创建 Ort::Value
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            data.data(),
            data.size(),
            shape.data(),
            shape.size()
        );

        // 将 Ort::Value 添加到输出向量
        img_encoder_out.push_back(std::move(tensor));
    }
}


void TensorRTInference::allocateInputAndOutputMemory()
{
    cudaError_t cudaStatus = cudaSetDevice(this->gpuId);
    if(ImageEncoder == TRTtype)
    {
        std::vector<std::string> inputNames = {"image", "batch_size"};

        std::vector<nvinfer1::Dims> inputShapes = {
            {4, {1, 3, 1024, 1024}},  // image: [batch_size, channels, height, width]
            {1, {this->batch_size}}  // batch_size: [batch_size]
        };

        // 清空之前的输入内存
        for (void* ptr : inputData) {
            cudaFree(ptr);
        }
        inputData.clear();
        inputSizes.clear();

        // 为每个输入张量分配 GPU 内存
        for (size_t i = 0; i < inputNames.size(); ++i) {
            // 计算输入张量的大小
            size_t inputSize = 1;
            for (int j = 0; j < inputShapes[i].nbDims; ++j) {
                inputSize *= inputShapes[i].d[j];
            }
            inputSize *= sizeof(float); // 假设输入数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, inputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
            }

            // 保存分配的内存和大小
            inputData.push_back(gpuMemory);
            inputSizes.push_back(inputSize);
        }

        std::vector<std::string> outputNames = {
            "pix_feat",
            "high_res_feat0",
            "high_res_feat1",
            "vision_feats",
            "vision_pos_embed"
        };
        // 设置输出张量的形状
        std::vector<nvinfer1::Dims> outputShapes = {
            {4, {this->batch_size, 256, 64, 64}},
            {4, {this->batch_size, 32, 256, 256}},
            {4, {this->batch_size, 64, 256, 256}},
            {4, {this->batch_size, 256, 64, 64}},
            {3, {4096, this->batch_size, 256}},
        };

        // 为每个输出张量分配 GPU 内存
        for (size_t i = 0; i < outputNames.size(); ++i) {
            // 计算输出张量的大小
            size_t outputSize = 1;
            for (int j = 0; j < outputShapes[i].nbDims; ++j) {
                outputSize *= outputShapes[i].d[j];
            }
            outputSize *= sizeof(float); // 假设输出数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, outputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
            }

            // 保存分配的内存和大小
            outputData.push_back(gpuMemory);
            outputSizes.push_back(outputSize);
        }
    }
    else if (MemoryAttention == TRTtype) {
        // 输入张量名称列表
        std::vector<std::string> inputNames = {
            "current_vision_feat",
            "current_vision_pos_embed",
            "memory_0",
            "memory_1",
            "memory_pos_embed"
        };

        // 输入张量的形状
        std::vector<nvinfer1::Dims> inputShapes = {
            {4, {this->batch_size, 256, 64, 64}},         // current_vision_feat
            {3, {4096, this->batch_size, 256}},          // current_vision_pos_embed
            {3, {obj_size, this->batch_size, 256}},      // memory_0
            {5, {mem_feature_size, this->batch_size, 64, 64, 64}}, // memory_1
            {3, {obj_size * 4 + mem_feature_size * 4096, this->batch_size, 64}} // memory_pos_embed
        };

        // 清空之前的输入内存
        for (void* ptr : inputData) {
            cudaFree(ptr);
        }
        inputData.clear();
        inputSizes.clear();

        // 为每个输入张量分配 GPU 内存
        for (size_t i = 0; i < inputNames.size(); ++i) {
            // 计算输入张量的大小
            size_t inputSize = 1;
            for (int j = 0; j < inputShapes[i].nbDims; ++j) {
                inputSize *= inputShapes[i].d[j];
            }
            inputSize *= sizeof(float); // 假设输入数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, inputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
            }

            // 保存分配的内存和大小
            inputData.push_back(gpuMemory);
            inputSizes.push_back(inputSize);
        }

        // 输出张量名称
        const char* outputTensorName = "image_embed"; // 替换为你的输出张量名称
        nvinfer1::Dims outputDims = {4, {this->batch_size, 256, 64, 64}};
        size_t outputSize = 1;
        for (int i = 0; i < outputDims.nbDims; ++i) {
            if(i == 0)
            {
                outputSize *= this->batch_size;
            }
            else 
            {
                outputSize *= outputDims.d[i];
            }
        }
        outputSize *= sizeof(float); // 输出数据的总字节数
        void* gpuMemory = nullptr;
        cudaMalloc(&gpuMemory, outputSize);
        if (!gpuMemory) {
            throw std::runtime_error("Failed to allocate GPU memory for output memory attention");
        }

        // 保存分配的内存和大小
        outputData.push_back(gpuMemory);
        outputSizes.push_back(outputSize);
    }
    else if (ImageDecoderFirst == TRTtype) {
        // 输入张量名称列表
        std::vector<std::string> inputNames = {
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1"
        };

        // 输出张量名称列表
        std::vector<std::string> outputNames = {
            "object_score_logits",
            "low_res_multimasks",
            "high_res_multimasks", 
            "sam_output_tokens",
            "ious", 
            "low_res_masks", 
            "high_res_masks", 
            "sam_output_token"
        };

        // 清空之前的输入和输出内存
        for (void* ptr : inputData) {
            cudaFree(ptr);
        }
        inputData.clear();
        inputSizes.clear();

        for (void* ptr : outputData) {
            cudaFree(ptr);
        }
        outputData.clear();
        outputSizes.clear();

        // 设置输入张量的形状
        std::vector<nvinfer1::Dims> inputShapes = {
            {3, {this->batch_size, 2, 2}},  
            {2, {this->batch_size, 2}},
            {4, {this->batch_size, 256, 64, 64}}, 
            {4, {this->batch_size, 32, 256, 256}}, 
            {4, {this->batch_size, 64, 128, 128}}
        };

        // 为每个输入张量分配 GPU 内存
        for (size_t i = 0; i < inputNames.size(); ++i) {
            // 计算输入张量的大小
            size_t inputSize = 1;
            for (int j = 0; j < inputShapes[i].nbDims; ++j) {
                inputSize *= inputShapes[i].d[j];
            }
            inputSize *= sizeof(float); // 假设输入数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, inputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
            }

            // 保存分配的内存和大小
            inputData.push_back(gpuMemory);
            inputSizes.push_back(inputSize);
        }

        // 设置输出张量的形状
        std::vector<nvinfer1::Dims> outputShapes = {
            {2, {this->batch_size, 1}},                     // object_score_logits: [3, 1]
            {4, {this->batch_size, 3, 256, 256}},           // low_res_multimasks: [3, 3, 256, 256]
            {4, {this->batch_size, 3, 1024, 1024}},         // high_res_multimasks: [3, 3, 1024, 1024]
            {3, {this->batch_size, 3, 256}},                // sam_output_tokens: [3, 3, 256]
            {2, {this->batch_size, 3}},                     // ious: [3, 3]
            {4, {this->batch_size, 1, 256, 256}},           // low_res_masks: [3, 1, 256, 256]
            {4, {this->batch_size, 1, 1024, 1024}},         // high_res_masks: [3, 1, 1024, 1024]
            {2, {this->batch_size, 256}}                    // sam_output_token: [3, 256]
        };

        // 为每个输出张量分配 GPU 内存
        for (size_t i = 0; i < outputNames.size(); ++i) {
            // 计算输出张量的大小
            size_t outputSize = 1;
            for (int j = 0; j < outputShapes[i].nbDims; ++j) {
                outputSize *= outputShapes[i].d[j];
            }
            outputSize *= sizeof(float); // 假设输出数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, outputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
            }

            // 保存分配的内存和大小
            outputData.push_back(gpuMemory);
            outputSizes.push_back(outputSize);
        }
    }
    else if (ImageDecoderSecond == TRTtype) {
        // 输入张量名称列表
        std::vector<std::string> inputNames = {
            "object_score_logits", 
            "frame_size", 
            "low_res_masks", 
            "high_res_masks", 
            "sam_output_token"
        };

        // 输出张量名称列表
        std::vector<std::string> outputNames = {
            "obj_ptr", 
            "mask_for_mem", 
            "pred_mask"
        };

        // 清空之前的输入和输出内存
        for (void* ptr : inputData) {
            cudaFree(ptr);
        }
        inputData.clear();
        inputSizes.clear();

        for (void* ptr : outputData) {
            cudaFree(ptr);
        }
        outputData.clear();
        outputSizes.clear();

        // 设置输入张量的形状
        std::vector<nvinfer1::Dims> inputShapes = {
            {2, {this->batch_size, 1}},  
            {1, {2}},
            {4, {this->batch_size, 1, 256, 256}}, 
            {4, {this->batch_size, 1, 1024, 1024}}, 
            {2, {this->batch_size, 256}}
        };

        // 为每个输入张量分配 GPU 内存
        for (size_t i = 0; i < inputNames.size(); ++i) {
            // 计算输入张量的大小
            size_t inputSize = 1;
            for (int j = 0; j < inputShapes[i].nbDims; ++j) {
                inputSize *= inputShapes[i].d[j];
            }
            if (inputNames[i] == "frame_size") {
                inputSize *= sizeof(int); // 假设输入数据是 int 类型
            } else {
                inputSize *= sizeof(float); // 假设输入数据是 float 类型
            }

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, inputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
            }

            // 保存分配的内存和大小
            inputData.push_back(gpuMemory);
            inputSizes.push_back(inputSize);
        }

        // 设置输出张量的形状
        std::vector<nvinfer1::Dims> outputShapes = {
            {2, {this->batch_size, 256}},
            {4, {this->batch_size, 1, 1024, 1024}},
            {4, {this->batch_size, 1, 1080, 1920}},
        };

        // 为每个输出张量分配 GPU 内存
        for (size_t i = 0; i < outputNames.size(); ++i) {
            // 计算输出张量的大小
            size_t outputSize = 1;
            for (int j = 0; j < outputShapes[i].nbDims; ++j) {
                outputSize *= outputShapes[i].d[j];
            }
            outputSize *= sizeof(float); // 假设输出数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, outputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
            }

            // 保存分配的内存和大小
            outputData.push_back(gpuMemory);
            outputSizes.push_back(outputSize);
        }
    }
    else if (MemoryEncoder == TRTtype) {
        // 输入张量名称列表
        std::vector<std::string> inputNames = {
            "mask_for_mem",
            "pix_feat"
        };

        // 输出张量名称列表
        std::vector<std::string> outputNames = {
            "maskmem_features",
            "maskmem_pos_enc",
            "temporal_code"
        };

        // 清空之前的输入和输出内存
        for (void* ptr : inputData) {
            cudaFree(ptr);
        }
        inputData.clear();
        inputSizes.clear();

        for (void* ptr : outputData) {
            cudaFree(ptr);
        }
        outputData.clear();
        outputSizes.clear();

        // 设置输入张量的形状
        std::vector<nvinfer1::Dims> inputShapes = {
            {4, {this->batch_size, 1, 1024, 1024}},  
            {4, {this->batch_size, 256, 64, 64}},
        };

        // 为每个输入张量分配 GPU 内存
        for (size_t i = 0; i < inputNames.size(); ++i) {
            // 计算输入张量的大小
            size_t inputSize = 1;
            for (int j = 0; j < inputShapes[i].nbDims; ++j) {
                inputSize *= inputShapes[i].d[j];
            }
            inputSize *= sizeof(float); // 假设输入数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, inputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
            }

            // 保存分配的内存和大小
            inputData.push_back(gpuMemory);
            inputSizes.push_back(inputSize);
            device_buffers.push_back(gpuMemory);
        }

        // 设置输出张量的形状
        std::vector<nvinfer1::Dims> outputShapes = {
            {4, {this->batch_size, 64, 64, 64}},                 
            {3, {4096, this->batch_size, 64}}, 
            {4, {7, 1, 1, 64}},         
        };

        // 为每个输出张量分配 GPU 内存
        for (size_t i = 0; i < outputNames.size(); ++i) {
            // 计算输出张量的大小
            size_t outputSize = 1;
            for (int j = 0; j < outputShapes[i].nbDims; ++j) {
                outputSize *= outputShapes[i].d[j];
            }
            outputSize *= sizeof(float); // 假设输出数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, outputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
            }

            // 保存分配的内存和大小
            outputData.push_back(gpuMemory);
            device_buffers.push_back(gpuMemory);
            outputSizes.push_back(outputSize);
        }
    }
    else if (ImageDecoder == TRTtype)
    {
        // 输入张量名称列表
        std::vector<std::string> inputNames = {
            "point_coords",
            "point_labels",
            "image_embed",
            "high_res_feats_0",
            "high_res_feats_1"
        };

        // 输出张量名称列表
        std::vector<std::string> outputNames = {
            "obj_ptr", 
            "mask_for_mem", 
            "pred_mask",
            "ious"
        };

        // 清空之前的输入和输出内存
        for (void* ptr : inputData) {
            cudaFree(ptr);
        }
        inputData.clear();
        inputSizes.clear();

        for (void* ptr : outputData) {
            cudaFree(ptr);
        }
        outputData.clear();
        outputSizes.clear();

        // 设置输入张量的形状
        std::vector<nvinfer1::Dims> inputShapes = {
            {3, {this->batch_size, 2, 2}},  
            {2, {this->batch_size, 2}},
            {4, {this->batch_size, 256, 64, 64}}, 
            {4, {this->batch_size, 32, 256, 256}}, 
            {4, {this->batch_size, 64, 128, 128}}
        };

        // 为每个输入张量分配 GPU 内存
        for (size_t i = 0; i < inputNames.size(); ++i) {
            // 计算输入张量的大小
            size_t inputSize = 1;
            for (int j = 0; j < inputShapes[i].nbDims; ++j) {
                inputSize *= inputShapes[i].d[j];
            }
            inputSize *= sizeof(float); // 假设输入数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, inputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for input " + inputNames[i]);
            }

            // 保存分配的内存和大小
            inputData.push_back(gpuMemory);
            inputSizes.push_back(inputSize);
        }

        // 设置输出张量的形状
        std::vector<nvinfer1::Dims> outputShapes = {
            {2, {this->batch_size, 256}},
            {4, {this->batch_size, 1, 1024, 1024}},
            {4, {this->batch_size, 1, 1080, 1920}},
            {2, {this->batch_size, 3}}
        };

        // 为每个输出张量分配 GPU 内存
        for (size_t i = 0; i < outputNames.size(); ++i) {
            // 计算输出张量的大小
            size_t outputSize = 1;
            for (int j = 0; j < outputShapes[i].nbDims; ++j) {
                outputSize *= outputShapes[i].d[j];
            }
            outputSize *= sizeof(float); // 假设输出数据是 float 类型

            // 分配 GPU 内存
            void* gpuMemory = nullptr;
            cudaMalloc(&gpuMemory, outputSize);
            if (!gpuMemory) {
                throw std::runtime_error("Failed to allocate GPU memory for output " + outputNames[i]);
            }

            // 保存分配的内存和大小
            outputData.push_back(gpuMemory);
            outputSizes.push_back(outputSize);
        }
    }
    cudaStreamCreate(&stream);
}

void Sam2Singleton::sam2Process(std::vector<cv::Mat> &images, cv::Rect& rectInfoIn, std::vector<cv::Rect>& rectOut)
{
    // if (images.empty()) {
    //     std::cout << "Sam2Singleton Input no image" << std::endl;
    //     return;
    // }

    // /* debug */
    // images.clear();
    // std::string video_path = "/home/l1/source_project_ywd/MOT/sam2_cpp/test/4.mp4";
    // cv::VideoCapture capture(video_path);
    // if (!capture.isOpened()) return;
    // cv::VideoWriter filesavewriter;
    // int idx = 0;
    // cv::Mat frame;
    // while (true) {

    //     if (!capture.read(frame) || frame.empty()) break;

    //     if(idx == 10)
    //        break;
    //     images.push_back(frame.clone());
    //     idx++;
    // }
    // /* debug */

    //clear old 记忆帧
    this->infer_status.current_frame = 0;
    this->infer_status.obj_ptr_first.clear();
    this->infer_status.status_first.clear();
    this->infer_status.status_recent.clear();
    this->infer_status.obj_ptr_recent.clear();
    this->infer_status.last_memoryFeature.clear();
    this->infer_status.last_objPtr.clear();
    this->parms.clear();
    this->LastRect = cv::Rect();

    std::vector<ParamsSam2> parms1;
    parms1.push_back({
        0,
        rectInfoIn,
        {0, 0}
    });

    setparms_InSam2Process(parms1);
    // setparms(parms1);
    auto startTime = std::chrono::high_resolution_clock::now();
    auto endTime   = std::chrono::high_resolution_clock::now();

    int image_cnt = 0;
    for(auto & image : images)
    {
        startTime = std::chrono::high_resolution_clock::now();
        this->inference(image);
        endTime   = std::chrono::high_resolution_clock::now();
        std::cerr << "Sam2Singleton cost: "<<std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count() << "ms" << std::endl;

        rectOut.push_back(this->LastRect);
        if(image_cnt == 0)
        {
            // cv::Mat roiImage = image(rectInfoIn);
            // if(!roiImage.empty())
            // {
            //     cv::imwrite("./sam2_out/imageStart.jpg", roiImage);
            // }
            cv::rectangle(image, rectInfoIn, cv::Scalar(0, 0, 255), 10);
            cv::imwrite("imageStart.jpg", image);
        }
        else if(image_cnt == images.size() - 1)
        {
            // cv::Mat roiImage = image(LastRect);

            // if(!roiImage.empty())
            // {
            //     cv::imwrite("./sam2_out/imageEnd.jpg", roiImage);
            // }
            cv::rectangle(image, this->LastRect, cv::Scalar(0, 0, 255), 10);
            cv::imwrite("imageEnd.jpg", image);
        }

        image_cnt++;
    }
}

int Sam2Singleton::setparms_InSam2Process(std::vector<ParamsSam2>& parms){
    this->parms = parms;
    infer_status.status_recent.resize(this->batch_size);
    infer_status.obj_ptr_recent.resize(this->batch_size);
    return 1;
}

int Sam2Singleton::setparms_AllocateBatch_One(){
    low_res_masks_data.resize(this->batch_size * 256 * 256, 0.0f);  // 调整大小并填充为 0
    high_res_masks_data.resize(this->batch_size * 1024 * 1024, 0.0f);  // 调整大小并填充为 0
    sam_output_data.resize(this->batch_size * 256, 0.0f);  // 调整大小并填充为 0
    infer_status.status_recent.resize(this->batch_size);
    infer_status.obj_ptr_recent.resize(this->batch_size);

    tensorrt_mem_attention_data.resize(this->batch_size * 256 * 64 * 64, 0.0f);

    mem_attention_engine_mem7_obj16->batch_size = this->batch_size;
    // ImageDecoderFirst_engine->batch_size = this->batch_size;
    ImageEndocer_engine->batch_size = this->batch_size;
    // ImageDecoderEnd_engine->batch_size = this->batch_size;
    MemoryEncoder_engine->batch_size = this->batch_size;
    ImageDecoder_engine->batch_size = this->batch_size;

    /* 开辟内存 */
    mem_attention_engine_mem7_obj16->TRTtype = MemoryAttention;
    // ImageDecoderFirst_engine->TRTtype = ImageDecoderFirst;
    ImageEndocer_engine->TRTtype = ImageEncoder;
    // ImageDecoderEnd_engine->TRTtype = ImageDecoderSecond;
    MemoryEncoder_engine->TRTtype = MemoryEncoder;
    ImageDecoder_engine->TRTtype = ImageDecoder;

    mem_attention_engine_mem7_obj16->allocateInputAndOutputMemory();
    // ImageDecoderFirst_engine->allocateInputAndOutputMemory();
    ImageEndocer_engine->allocateInputAndOutputMemory();
    // ImageDecoderEnd_engine->allocateInputAndOutputMemory();
    MemoryEncoder_engine->allocateInputAndOutputMemory();
    ImageDecoder_engine->allocateInputAndOutputMemory();

    outputHostData.resize(5);
    outputHostData[0].resize(this->batch_size * 256 * 64 * 64, 0.0f);
    outputHostData[1].resize(this->batch_size * 32 * 256 * 256, 0.0f);
    outputHostData[2].resize(this->batch_size * 64 * 256 * 256, 0.0f);
    outputHostData[3].resize(this->batch_size * 256 * 64 * 64, 0.0f);
    outputHostData[4].resize(this->batch_size * 4096 * 256, 0.0f);

    outputDataForImageDecoder_First.resize(8);
    outputDataForImageDecoder_First[0].resize(this->batch_size * 1, 0.0f);
    outputDataForImageDecoder_First[1].resize(this->batch_size * 3 * 256 * 256, 0.0f);
    outputDataForImageDecoder_First[2].resize(this->batch_size * 3 * 1024 * 1024, 0.0f);
    outputDataForImageDecoder_First[3].resize(this->batch_size * 3 * 256, 0.0f);
    outputDataForImageDecoder_First[4].resize(this->batch_size * 3, 0.0f);
    outputDataForImageDecoder_First[5].resize(this->batch_size * 1 * 256 * 256, 0.0f);
    outputDataForImageDecoder_First[6].resize(this->batch_size * 1 * 1024 * 1024, 0.0f);
    outputDataForImageDecoder_First[7].resize(this->batch_size * 256, 0.0f);

    outputDataForImageDecoder_Second.resize(4);
    outputDataForImageDecoder_Second[0].resize(this->batch_size * 256, 0.0f);
    outputDataForImageDecoder_Second[1].resize(this->batch_size * 1 * 1024 * 1024, 0.0f);
    outputDataForImageDecoder_Second[2].resize(this->batch_size * 1 * 1080 * 1920, 0.0f);   //todoYWD 1080 and 1920
    outputDataForImageDecoder_Second[3].resize(this->batch_size * 3, 0.0f);

    outputDataForMmeEncoder.resize(3);
    outputDataForMmeEncoder[0].resize(this->batch_size * 64 * 64 * 64, 0.0f);
    outputDataForMmeEncoder[1].resize(4096 * this->batch_size * 64, 0.0f);
    outputDataForMmeEncoder[2].resize(7 * 1 * 1 * 64, 0.0f);

    image_preprocess_data.resize(1 * 3 * 1024 * 1024);

    gpu_buffers.resize(5);

    channel_buffers.resize(3);

    convert8UC1_mat = cv::Mat(1080, 1920, CV_8UC1);

    obj_ptrs.resize(16 * 256 * this->batch_size);    //固定为16

    maskmem_features_.resize(7*64*64*64*this->batch_size); //固定为7

    maskmem_pos_enc_.resize((7*4096+4*16)*this->batch_size*64);
    return 1;
}

std::vector<float> Sam2Singleton::get_best_iou()
{
    std::vector<float> best_iou;
    const float* iou_data = outputDataForImageDecoder_Second[3].data();

    for (int i = 0; i < this->batch_size; ++i)
    {
        float iou1 = iou_data[i * 3 + 0];
        float iou2 = iou_data[i * 3 + 1];
        float iou3 = iou_data[i * 3 + 2];

        float max_iou = std::max({iou1, iou2, iou3});
        best_iou.push_back(max_iou);
    }

    return best_iou;
}

}
