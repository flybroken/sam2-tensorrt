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

// ---- Demo 1: 单图片推理 ----
static void demo_single_image(
    const std::string& image_path,
    const cv::Rect&    init_bbox)
{
    std::cout << "\n===== Demo: Single Image =====\n";
    auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

    std::vector<std::string> engine_paths = {
        "models/memory_attention.engine",
        "models/image_encoder.engine",
        "models/image_decoderStart.engine",
        "models/image_decoderEnd.engine",
        "models/memory_encoder.engine",
        "models/image_decoder.engine"
    };

    sam2.initialize(engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::SingleTrack, 0);

    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return;
    }

    std::vector<cv::Mat> images;
    images.push_back(image);

    std::vector<cv::Rect> rect_out;
    sam2.sam2Process(images, const_cast<cv::Rect&>(init_bbox), rect_out);

    std::cout << "Output bbox: " << rect_out[0] << std::endl;
    std::cout << "Output image saved as: imageStart.jpg / imageEnd.jpg / result.png\n";
}

// ---- Demo 2: 视频逐帧推理 ----
static void demo_video(
    const std::string& video_path,
    const cv::Rect&    init_bbox,
    int                max_frames)
{
    std::cout << "\n===== Demo: Video Tracking =====\n";

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video: " << video_path << std::endl;
        return;
    }
    print_video_info(cap);

    // 读取前若干帧到内存
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (frames.size() < static_cast<size_t>(max_frames)) {
        if (!cap.read(frame) || frame.empty()) break;
        frames.push_back(frame.clone());
    }
    cap.release();
    std::cout << "Loaded " << frames.size() << " frames.\n";

    auto& sam2 = TrackerBySAM2::Sam2Singleton::getInstance();

    std::vector<std::string> engine_paths = {
        "models/memory_attention.engine",
        "models/image_encoder.engine",
        "models/image_decoderStart.engine",
        "models/image_decoderEnd.engine",
        "models/memory_encoder.engine",
        "models/image_decoder.engine"
    };

    sam2.initialize(engine_paths, TrackerBySAM2::TRACKTYPEBYSAM2::SingleTrack, 0);

    std::vector<cv::Rect> rect_out;
    auto t_start = std::chrono::high_resolution_clock::now();
    sam2.sam2Process(frames, const_cast<cv::Rect&>(init_bbox), rect_out);
    auto t_end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    double avg_ms   = total_ms / frames.size();
    double avg_fps  = 1000.0 / avg_ms;

    std::cout << "\n--- Benchmark ---\n";
    std::cout << "Frames processed: " << frames.size() << "\n";
    std::cout << "Total time:       " << total_ms << " ms\n";
    std::cout << "Avg per frame:    " << avg_ms   << " ms\n";
    std::cout << "Avg FPS:         " << avg_fps  << "\n";
    std::cout << "Initial bbox:     " << init_bbox << "\n";
    std::cout << "Final bbox:       " << rect_out.back() << "\n";
    std::cout << "Output saved as:  imageStart.jpg / imageEnd.jpg / result.png\n";
}

// ---- 用法说明 ----
static void print_usage(const char* prog) {
    std::cout << "Usage:\n"
              << "  " << prog << " <mode> <path> <x> <y> <w> <h> [max_frames]\n\n"
              << "Modes:\n"
              << "  image   Single image inference\n"
              << "  video   Video tracking with benchmark\n\n"
              << "Examples:\n"
              << "  " << prog << " image test/image.jpg 100 200 150 300\n"
              << "  " << prog << " video test/input.mp4 100 200 150 300 50\n";
}

int main(int argc, char** argv) {
    if (argc < 7) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode       = argv[1];
    std::string media_path = argv[2];
    cv::Rect   init_bbox(
        std::stoi(argv[3]),  // x
        std::stoi(argv[4]),  // y
        std::stoi(argv[5]),  // width
        std::stoi(argv[6])   // height
    );

    std::cout << "Mode: " << mode << ", Path: " << media_path
              << ", Init bbox: " << init_bbox << std::endl;

    if (mode == "image") {
        demo_single_image(media_path, init_bbox);
    } else if (mode == "video") {
        int max_frames = (argc >= 8) ? std::stoi(argv[7]) : 50;
        demo_video(media_path, init_bbox, max_frames);
    } else {
        std::cerr << "Unknown mode: " << mode << "\n";
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
