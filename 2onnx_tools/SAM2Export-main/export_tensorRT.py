import tensorrt as trt
import os
import ctypes
import onnx
import torch




class MemattenionCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, calibration_data, batch_size=1):
        """
        初始化校准器。

        参数:
            calibration_data: 校准数据，是一个包含输入张量的字典。
            batch_size: 每个校准批次的大小。
        """
        super().__init__()
        self.calibration_data = calibration_data
        self.batch_size = batch_size
        self.current_index = 0

        # 为每个输入张量分配 GPU 内存
        self.device_buffers = {}
        for name, data in self.calibration_data.items():
            self.device_buffers[name] = self._allocate_device_memory(data)

    def _allocate_device_memory(self, data):
        """
        为输入数据分配 GPU 内存。
        """
        data_np = data.numpy()  # 将 PyTorch 张量转换为 NumPy 数组
        data_gpu = torch.from_numpy(data_np).cuda()  # 使用 PyTorch 将数据移动到 GPU
        return int(data_gpu.data_ptr())  # 返回 GPU 内存指针

    def get_batch_size(self):
        """
        返回每个校准批次的大小。
        """
        return self.batch_size

    def get_batch(self, names):
        """
        获取下一个校准批次的数据。
        """
        if self.current_index >= len(self.calibration_data):
            return None  # 校准数据已用完

        # 返回当前批次的 GPU 内存指针
        batch = []
        for name in names:
            batch.append(self.device_buffers[name])
        self.current_index += 1
        return batch

    def read_calibration_cache(self):
        """
        读取校准缓存（如果存在）。
        """
        return None  # 不读取缓存，每次重新校准

    def write_calibration_cache(self, cache):
        """
        写入校准缓存。
        """
        pass  # 不保存缓存

    def __del__(self):
        """
        释放 GPU 内存。
        """
        for buffer in self.device_buffers.values():
            torch.cuda.Free(int(buffer))  # 释放 GPU 内存







def generate_calibration_data(num_samples=50):
    """
    生成校准数据。

    参数:
        num_samples: 校准数据的样本数量。
    """
    calibration_data = {
        "current_vision_feat": torch.randn(num_samples, 256, 64, 64),  # [num_samples, 256, 64, 64]
        "current_vision_pos_embed": torch.randn(num_samples, 4096, 1, 256),  # [num_samples, 4096, 1, 256]
        "memory_0": torch.randn(num_samples, 16, 1, 256),  # [num_samples, 16, 1, 256]
        "memory_1": torch.randn(num_samples, 7, 1, 64, 64, 64),  # [num_samples, 7, 1, 64, 64, 64]
        "memory_pos_embed": torch.randn(num_samples, 7 * 4096 + 4 * 16, 1, 64),  # [num_samples, y*4096, 1, 64]
    }
    return calibration_data


def build_engine_with_int8(network, builder, config, calibration_data):
    """
    使用 INT8 校准器构建 TensorRT 引擎。
    """
    if builder.platform_has_fast_int8:
        print("Platform supports INT8.")
        config.set_flag(trt.BuilderFlag.INT8)  # 启用 INT8 模式

        # 创建校准器
        calibrator = MemattenionCalibrator(calibration_data)
        config.int8_calibrator = calibrator
    else:
        print("Platform does not support INT8.")
        config.set_flag(trt.BuilderFlag.FP16)  # 如果不支持 INT8，启用 FP16 模式

    # 构建引擎
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")
    return engine


def export_MemAttention_tensorrt_engine(onnx_path, engine_path):
    # 初始化TensorRT组件
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)

    # 加载插件库
    ctypes.CDLL("/shared/ywd/MultiTracker/yanwendou/export_sam2_model/sam2_cpp/2onnx_tools/SAM2Export-main/TensorRT-10.0.0.6/lib/libnvinfer_plugin.so")
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 加载ONNX模型
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 13 << 30)  # 1GB

    # 设置动态形状配置文件
    profile = builder.create_optimization_profile()
    
    # 配置每个输入的动态维度
    # current_vision_feat: [batch_size_attention1, 256, 64, 64]
    profile.set_shape(
        "current_vision_feat",
        min=(1, 256, 64, 64),
        opt=(3, 256, 64, 64),  # 使用示例中的典型值
        max=(10, 256, 64, 64)
    )

    # current_vision_pos_embed: [batch_size_attention2, batch_size_attention6, 256]
    profile.set_shape(
        "current_vision_pos_embed",
        min=(4096, 1, 256),
        opt=(4096, 3, 256),
        max=(4096, 10, 256)
    )
    
    # memory_0: [num, batch_size_attention3]
    profile.set_shape(
        "memory_0",
        min=(16, 1, 256),
        opt=(16, 3, 256),
        max=(16, 10, 256)
    )
    
    # memory_1: [buff_size, batch_size_attention4, 64, 64, 64]
    profile.set_shape(
        "memory_1",
        min=(7, 1, 64, 64, 64),
        opt=(7, 3, 64, 64, 64),
        max=(7, 10, 64, 64, 64)  
    )
    
    # memory_pos_embed: [buff_size, batch_size_attention5, 64]
    profile.set_shape(
        "memory_pos_embed",
        min=(4096 * 7 + 4 * 16, 1,  64),
        opt=(4096 * 7 + 4 * 16, 3,  64),
        max=(4096 * 7 + 4 * 16, 10, 64)
    )

    config.add_optimization_profile(profile)

    # 设置精度（可选）
    if builder.platform_has_fast_int8:
        print("Platform supports INT8.")
        config.set_flag(trt.BuilderFlag.INT8)  # 启用INT8模式

        # 使用校准工具生成动态范围
        calibrator = MemattenionCalibrator()  # 自定义校准器
        config.int8_calibrator = calibrator


    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)

    if builder.platform_has_fast_int8:
        print("Platform supports INT8.")
    else:
        print("Platform does not support INT8.")

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")


def export_ImageEncoder_tensorrt_engine(onnx_path, engine_path):
    """
    将图像编码器的 ONNX 模型导出为 TensorRT 引擎，并启用 FP16 精度。

    参数:
        onnx_path (str): ONNX 模型文件路径。
        engine_path (str): 导出的 TensorRT 引擎文件路径。
    """
    # 初始化 TensorRT 日志记录器
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建 TensorRT 构建器和网络
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 加载 ONNX 模型
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)  # 1GB 工作空间

    # 设置动态形状配置文件
    profile = builder.create_optimization_profile()

    # 配置输入动态维度
    profile.set_shape(
        "image",  # 输入名称
        min=(1, 3, 1024, 1024),  # 最小输入形状
        opt=(1, 3, 1024, 1024),  # 最优输入形状
        max=(1, 3, 1024, 1024)   # 最大输入形状
    )

    profile.set_shape(
        "batch_size",  # 输入名称
        min=(1,),  # 最小输入形状
        opt=(5,),  # 最优输入形状
        max=(10,)  # 最大输入形状
    )

    config.add_optimization_profile(profile)

    # 启用 FP16 精度
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")


def export_image_decoderStart_tensorrt_engine(onnx_path, engine_path):
    """
    将图像解码器的 ONNX 模型导出为 TensorRT 引擎，并启用 FP16 精度。

    参数:
        onnx_path (str): ONNX 模型文件路径。
        engine_path (str): 导出的 TensorRT 引擎文件路径。
    """
    # 初始化 TensorRT 日志记录器
    logger = trt.Logger(trt.Logger.WARNING)
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    # 创建 TensorRT 构建器和网络
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    ctypes.CDLL("/shared/ywd/MultiTracker/yanwendou/export_sam2_model/sam2_cpp/2onnx_tools/SAM2Export-main/TensorRT-10.8.0.43/lib/libnvinfer_plugin.so")
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')
    # 加载 ONNX 模型
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)  # 5GB 工作空间

    # 设置动态形状配置文件
    profile = builder.create_optimization_profile()

    # 配置输入动态维度
    profile.set_shape(
        "point_coords",  # 输入名称
        min=(1, 2, 2),  # 最小输入形状
        opt=(2, 2, 2),  # 最优输入形状
        max=(10, 2, 2)  # 最大输入形状
    )

    profile.set_shape(
        "point_labels",  # 输入名称
        min=(1, 2),  # 最小输入形状
        opt=(2, 2),  # 最优输入形状
        max=(10, 2)  # 最大输入形状
    )

    profile.set_shape(
        "image_embed",  # 输入名称
        min=(1, 256, 64, 64),  # 最小输入形状
        opt=(2, 256, 64, 64),  # 最优输入形状
        max=(10, 256, 64, 64)  # 最大输入形状
    )

    profile.set_shape(
        "high_res_feats_0",  # 输入名称
        min=(1, 32, 256, 256),  # 最小输入形状
        opt=(2, 32, 256, 256),  # 最优输入形状
        max=(10, 32, 256, 256)  # 最大输入形状
    )

    profile.set_shape(
        "high_res_feats_1",  # 输入名称
        min=(1, 64, 128, 128),  # 最小输入形状
        opt=(2, 64, 128, 128),  # 最优输入形状
        max=(10, 64, 128, 128)  # 最大输入形状
    )

    config.add_optimization_profile(profile)

    # 启用 FP16 精度
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")


def export_image_decoderEnd_tensorrt_engine(onnx_path, engine_path):
    """
    将图像解码器的 ONNX 模型导出为 TensorRT 引擎，并启用 FP16 精度。

    参数:
        onnx_path (str): ONNX 模型文件路径。
        engine_path (str): 导出的 TensorRT 引擎文件路径。
    """
    # 初始化 TensorRT 日志记录器
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建 TensorRT 构建器和网络
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # 加载 ONNX 模型
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)  # 5GB 工作空间

    # 设置动态形状配置文件
    profile = builder.create_optimization_profile()

    # 配置输入动态维度
    profile.set_shape(
        "object_score_logits",  # 输入名称
        min=(1, 1),  # 最小输入形状
        opt=(3, 1),  # 最优输入形状
        max=(10, 1)  # 最大输入形状
    )

    profile.set_shape(
        "frame_size",  # 输入名称
        min=(2,),  # 最小输入形状
        opt=(2,),  # 最优输入形状
        max=(2,)  # 最大输入形状
    )

    profile.set_shape(
        "low_res_masks",  # 输入名称
        min=(1, 1, 256, 256),  # 最小输入形状
        opt=(3, 1, 256, 256),  # 最优输入形状
        max=(10, 1, 256, 256)  # 最大输入形状
    )

    profile.set_shape(
        "high_res_masks",  # 输入名称
        min=(1, 1, 1024, 1024),  # 最小输入形状
        opt=(3, 1, 1024, 1024),  # 最优输入形状
        max=(10, 1, 1024, 1024)  # 最大输入形状
    )

    profile.set_shape(
        "sam_output_token",  # 输入名称
        min=(1, 256),  # 最小输入形状
        opt=(3, 256),  # 最优输入形状
        max=(10, 256)  # 最大输入形状
    )

    config.add_optimization_profile(profile)

    # 启用 FP16 精度
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)

    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")

def export_memory_encoder_tensorrt_engine(onnx_path, engine_path):
    """
    将内存编码器的 ONNX 模型导出为 TensorRT 引擎，并禁用优化，强制使用 FP32 精度。

    参数:
        onnx_path (str): ONNX 模型文件路径。
        engine_path (str): 导出的 TensorRT 引擎文件路径。
    """
    # 初始化 TensorRT 日志记录器
    logger = trt.Logger(trt.Logger.WARNING)

    # 创建 TensorRT 构建器和网络
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    ctypes.CDLL("/shared/ywd/MultiTracker/yanwendou/export_sam2_model/sam2_cpp/2onnx_tools/SAM2Export-main/TensorRT-10.8.0.43/lib/libnvinfer_plugin.so")
    trt.init_libnvinfer_plugins(TRT_LOGGER, '')

    # 加载 ONNX 模型
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            print("Failed parsing ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return

    # 配置构建器
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 5 << 30)  # 5GB 工作空间

    # 设置动态形状配置文件
    profile = builder.create_optimization_profile()

    # 配置输入动态维度
    profile.set_shape(
        "mask_for_mem",  # 输入名称
        min=(1, 1, 1024, 1024),  # 最小输入形状
        opt=(3, 1, 1024, 1024),  # 最优输入形状
        max=(10, 1, 1024, 1024)  # 最大输入形状
    )

    profile.set_shape(
        "pix_feat",  # 输入名称
        min=(1, 256, 64, 64),  # 最小输入形状
        opt=(3, 256, 64, 64),  # 最优输入形状
        max=(10, 256, 64, 64)  # 最大输入形状
    )

    config.add_optimization_profile(profile)
    config.set_flag(trt.BuilderFlag.FP16)


    # 构建引擎
    serialized_engine = builder.build_serialized_network(network, config)

    # 保存引擎文件
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    print(f"TensorRT engine saved to {engine_path}")

# 使用示例
if __name__ == "__main__":
    # # #onnx_model_path = "/shared/ywd/MultiTracker/yanwendou/export_sam2_model/sam2_cpp/2onnx_tools/SAM2Export-main/checkpoints/base_plus/memory_attention.onnx"
    # # trt_engine_path = "memory_attention7_16.engine"

    # # onnx_model_path = "memory_attention_modified_final7_16.onnx"
    # # # 先确保ONNX模型存在
    # # if not os.path.exists(onnx_model_path):
    # #     raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    # # model = onnx.load(onnx_model_path)

    # # nodes = model.graph.node

    # # # ONNX 节点索引通常从0开始，第13个节点是 nodes[12]
    # # node = nodes[13]  # 调整索引

    # # export_MemAttention_tensorrt_engine(onnx_model_path, trt_engine_path)

    # # # ONNX 模型路径
    # # onnx_model_path = "/shared/ywd/MultiTracker/yanwendou/export_sam2_model/sam2_cpp/2onnx_tools/SAM2Export-main/checkpoints/base_plus/image_encoder.onnx"
    # # # TensorRT 引擎保存路径
    # # trt_engine_path = "image_encoder_fp16.engine"

    # # # 确保 ONNX 模型存在
    # # if not os.path.exists(onnx_model_path):
    # #     raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    # # # 导出 TensorRT 引擎
    # # export_ImageEncoder_tensorrt_engine(onnx_model_path, trt_engine_path)

    # # ONNX 模型路径
    # onnx_model_path = "memory_attention_modified_final7_16.onnx"
    # # TensorRT 引擎保存路径
    # trt_engine_path = "memory_attention_int8.engine"

    # # 确保 ONNX 模型存在
    # if not os.path.exists(onnx_model_path):
    #     raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

    # # 导出 TensorRT 引擎
    # export_MemAttention_tensorrt_engine(onnx_model_path, trt_engine_path)

    calibration_data = generate_calibration_data(num_samples=100)

    # 创建 TensorRT 构建器和网络
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()

    # 解析 ONNX 模型
    parser = trt.OnnxParser(network, logger)
    with open("memory_attention_modified_final7_16.onnx", "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model.")

    # 使用 INT8 校准器构建引擎
    engine = build_engine_with_int8(network, builder, config, calibration_data)

    # 保存引擎
    with open("memory_attention7_16_int8.engine", "wb") as f:
        f.write(engine.serialize())

    print("INT8 engine built and saved successfully.")