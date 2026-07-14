#include <iostream>
#include <chrono>
#include <vector>
#include <opencv2/opencv.hpp>
#include "SAM2.h"

// ---- 辅助：打印视频信息 ----
static void print_video_info(const cv::VideoCapture& cap) {
    std::cout << "Video width:  " << cap.get(cv::CAP_PROP_FRAME_WIDTH)  << std::endl;
    std::cout << "Video height: " << cap.get(cv::CAP_PROP_FRAME_HEIGHT) << std::endl;
    std::cout << "Video FPS:    " << cap.get(cv::CAP_PROP_FPS)          << std::endl;
    std::cout << "Video frames: " << cap.get(cv::CAP_PROP_FRAME_COUNT)  << std::endl;
}

// 引擎路径（所有 demo 共用）
static std::vector<std::string> singleObj_engine_paths = {
    "models/fp16_small_singleObj/memory_attention7_16.engine",
    "models/fp16_small_singleObj/image_encoder.engine",
    "models/fp16_small_singleObj/image_decoderStart.engine",
    "models/fp16_small_singleObj/image_decoderEnd.engine",
    "models/fp16_small_singleObj/memory_encoder.engine",
    "models/fp16_small_singleObj/image_decoder.engine"
};

static std::vector<std::string> multiObj_engine_paths = {
    "models/fp16_small_motObj/memory_attention7_16.engine",
    "models/fp16_small_motObj/image_encoder.engine",
    "models/fp16_small_motObj/image_decoderStart.engine",
    "models/fp16_small_motObj/image_decoderEnd.engine",
    "models/fp16_small_motObj/memory_encoder.engine",
    "models/fp16_small_motObj/image_decoder.engine"
};

// ---- Demo 1: 单图片推理 (SingleTrack) ----
static void demo_single_image(
    const std::string& image_path,
    const cv::Rect&    init_bbox)
{
    std::cout << "\n===== Demo: Single Image (initialize → setparms → inference) =====\n";
    auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

    // step1: initialize 构建引擎
    sam2.initialize(singleObj_engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::SingleTrack, 0);

    // step2: setparms 设定单目标 + 分配 batch=1 内存
    std::vector<TrackerBySAM2::ParamsSam2> parms;
    parms.push_back({0, init_bbox, {0, 0}});
    sam2.setparms(parms);

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }

    // step3: inference 单帧推理
    sam2.inference(image);
    cv::Rect result = sam2.LastRect;
    std::cout << "Output bbox: " << result << std::endl;

    // 保存结果
    cv::Mat img_start = cv::imread(image_path);
    cv::rectangle(img_start, init_bbox, cv::Scalar(0, 0, 255), 10);
    cv::imwrite("imageStart.jpg", img_start);

    cv::Mat img_end = cv::imread(image_path);
    cv::rectangle(img_end, result, cv::Scalar(0, 0, 255), 10);
    cv::imwrite("imageEnd.jpg", img_end);
    std::cout << "Output saved as: imageStart.jpg / imageEnd.jpg\n";
}

// ---- Demo 2: 视频单目标追踪 (SingleTrack) ----
static void demo_video(
    const std::string& video_path,
    const cv::Rect&    init_bbox,
    int                max_frames)
{
    std::cout << "\n===== Demo: Video SingleTrack (initialize → setparms → inference) =====\n";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return;
    }
    print_video_info(cap);

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (frames.size() < static_cast<size_t>(max_frames)) {
        if (!cap.read(frame) || frame.empty()) break;
        frames.push_back(frame.clone());
    }
    cap.release();
    std::cout << "Loaded " << frames.size() << " frames.\n";

    auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

    // step1: initialize 构建引擎
    sam2.initialize(singleObj_engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::SingleTrack, 0);

    // step2: setparms 设定单目标 + 分配 batch=1 内存
    std::vector<TrackerBySAM2::ParamsSam2> parms;
    parms.push_back({0, init_bbox, {0, 0}});
    sam2.setparms(parms);

    // step3: 逐帧 inference + benchmark + 视频输出
    std::vector<cv::Rect> rect_out;
    cv::VideoWriter writer("single_track_output.avi",
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           30, frames[0].size());
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer.\n";
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < frames.size(); ++i) {
        sam2.inference(frames[i]);
        rect_out.push_back(sam2.LastRect);

        // 每帧画 bbox 并写入视频
        cv::Mat out_frame = frames[i].clone();
        cv::rectangle(out_frame, sam2.LastRect, cv::Scalar(0, 255, 0), 2);
        if (writer.isOpened()) {
            writer.write(out_frame);
        }

        if (i == 0) {
            cv::rectangle(out_frame, init_bbox, cv::Scalar(0, 0, 255), 2);
            cv::imwrite("imageStart.jpg", out_frame);
        }
        if (i == frames.size() - 1) {
            cv::imwrite("imageEnd.jpg", out_frame);
        }
    }

    writer.release();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms   = total_ms / frames.size();
    double avg_fps  = 1000.0 / avg_ms;

    std::cout << "\n--- Benchmark (SingleTrack) ---\n";
    std::cout << "Frames processed: " << frames.size() << "\n";
    std::cout << "Total time:       " << total_ms << " ms\n";
    std::cout << "Avg per frame:    " << avg_ms   << " ms\n";
    std::cout << "Avg FPS:         " << avg_fps  << "\n";
    std::cout << "Initial bbox:     " << init_bbox << "\n";
    std::cout << "Final bbox:       " << rect_out.back() << "\n";
    std::cout << "Output saved as:  imageStart.jpg / imageEnd.jpg / single_track_output.avi\n";
}

// ---- Demo 3: 视频多目标追踪 (MultiTrack) ----
static void demo_mot_video(
    const std::string& video_path,
    const std::vector<cv::Rect>& init_bboxes,
    int max_frames)
{
    std::cout << "\n===== Demo: Video MultiTrack (initialize → setparms → inference) =====\n";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return;
    }
    print_video_info(cap);

    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (frames.size() < static_cast<size_t>(max_frames)) {
        if (!cap.read(frame) || frame.empty()) break;
        frames.push_back(frame.clone());
    }
    cap.release();
    std::cout << "Loaded " << frames.size() << " frames.\n";

    auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

    // step1: initialize 构建引擎（MultiTrack 推迟内存分配到 setparms）
    sam2.initialize(multiObj_engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::MultiTrack, 0);

    // step2: setparms 设定多目标 + 分配 batch=N 内存
    std::vector<TrackerBySAM2::ParamsSam2> parms;
    for (const auto& bbox : init_bboxes) {
        parms.push_back({0, bbox, {0, 0}});
    }
    std::cout << "Tracking " << parms.size() << " targets.\n";
    sam2.setparms(parms);

    // step3: 逐帧 inference + benchmark + 视频输出
    cv::VideoWriter writer("multi_track_output.avi",
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           30, frames[0].size());
    if (!writer.isOpened()) {
        std::cerr << "Failed to open video writer.\n";
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    for (size_t i = 0; i < frames.size(); ++i) {
        sam2.inference(frames[i]);

        // 每帧画 bbox 并写入视频
        cv::Mat out_frame = frames[i].clone();
        // 初始框（红色）

        if (0 == i)
        {
            for (const auto& bbox : init_bboxes) {
                cv::rectangle(out_frame, bbox, cv::Scalar(0, 0, 255), 2);
            }
        }

        // // 当前追踪框（绿色）
        // cv::rectangle(out_frame, sam2.LastRect, cv::Scalar(0, 255, 0), 2);
        if (writer.isOpened()) {
            writer.write(out_frame);
        }

        if (i == 0) {
            cv::imwrite("imageStart.jpg", out_frame);
        }
        if (i == frames.size() - 1) {
            cv::imwrite("imageEnd.jpg", out_frame);
        }
    }

    writer.release();

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms   = total_ms / frames.size();
    double avg_fps  = 1000.0 / avg_ms;

    std::cout << "\n--- Benchmark (MultiTrack) ---\n";
    std::cout << "Targets:          " << init_bboxes.size() << "\n";
    std::cout << "Frames processed: " << frames.size() << "\n";
    std::cout << "Total time:       " << total_ms << " ms\n";
    std::cout << "Avg per frame:    " << avg_ms   << " ms\n";
    std::cout << "Avg FPS:         " << avg_fps  << "\n";
    std::cout << "Output saved as:  imageStart.jpg / imageEnd.jpg / multi_track_output.avi\n";
}

// ---- 用法说明 ----
static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " image <path> <x> <y> <w> <h>\n"
              << "  " << prog << " video <path> <x> <y> <w> <h> [max_frames]\n"
              << "  " << prog << " mot   <path> <x1> <y1> <w1> <h1> [x2 y2 w2 h2 ...] [max_frames]\n\n"
              << "Modes:\n"
              << "  image   Single image inference  (SingleTrack)\n"
              << "  video   Video single target     (SingleTrack)\n"
              << "  mot     Video multi-target      (MultiTrack)\n\n"
              << "Examples:\n"
              << "  " << prog << " image test/image.jpg 100 200 150 300\n"
              << "  " << prog << " video test/input.mp4 100 200 150 300 50\n"
              << "  " << prog << " mot   test/input.mp4 100 200 150 300 400 500 200 180\n";
}

int main(int argc, char** argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode       = argv[1];
    std::string media_path = argv[2];

    if (mode == "image") {
        cv::Rect init_bbox(
            std::stoi(argv[3]), std::stoi(argv[4]),
            std::stoi(argv[5]), std::stoi(argv[6])
        );
        std::cout << "Mode: " << mode << ", Path: " << media_path
                  << ", Init bbox: " << init_bbox << std::endl;
        demo_single_image(media_path, init_bbox);
    } else if (mode == "video") {
        cv::Rect init_bbox(
            std::stoi(argv[3]), std::stoi(argv[4]),
            std::stoi(argv[5]), std::stoi(argv[6])
        );
        int max_frames = (argc >= 8) ? std::stoi(argv[7]) : 50;
        std::cout << "Mode: " << mode << ", Path: " << media_path
                  << ", Init bbox: " << init_bbox
                  << ", Max frames: " << max_frames << std::endl;
        demo_video(media_path, init_bbox, max_frames);
    } else if (mode == "mot") {
        // 解析多个 bbox: x1 y1 w1 h1 [x2 y2 w2 h2 ...]
        std::vector<cv::Rect> init_bboxes;
        int arg_idx = 3;
        while (arg_idx + 3 < argc) {
            // 检查是否是最后一组 bbox（后面是 max_frames 或结束）
            // 如果剩余参数恰好 1 个且是整数，则当作 max_frames
            if (arg_idx + 4 == argc - 1) {
                // 最后一组 bbox + 1 个额外参数 → max_frames
                init_bboxes.emplace_back(
                    std::stoi(argv[arg_idx]), std::stoi(argv[arg_idx + 1]),
                    std::stoi(argv[arg_idx + 2]), std::stoi(argv[arg_idx + 3])
                );
                arg_idx += 4;
                break;
            }
            if (arg_idx + 4 == argc) {
                // 正好最后一组 bbox，无 max_frames
                init_bboxes.emplace_back(
                    std::stoi(argv[arg_idx]), std::stoi(argv[arg_idx + 1]),
                    std::stoi(argv[arg_idx + 2]), std::stoi(argv[arg_idx + 3])
                );
                arg_idx += 4;
                break;
            }
            if (arg_idx + 4 < argc) {
                init_bboxes.emplace_back(
                    std::stoi(argv[arg_idx]), std::stoi(argv[arg_idx + 1]),
                    std::stoi(argv[arg_idx + 2]), std::stoi(argv[arg_idx + 3])
                );
                arg_idx += 4;
            }
        }

        if (init_bboxes.empty()) {
            std::cerr << "mot mode requires at least one bbox (x y w h).\n";
            print_usage(argv[0]);
            return 1;
        }

        int max_frames = (arg_idx < argc) ? std::stoi(argv[arg_idx]) : 50;

        std::cout << "Mode: " << mode << ", Path: " << media_path
                  << ", Targets: " << init_bboxes.size()
                  << ", Max frames: " << max_frames << std::endl;
        demo_mot_video(media_path, init_bboxes, max_frames);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
