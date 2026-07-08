import torch
import onnx
import argparse
from onnxsim import simplify
from src.Module import ImageEncoder
from src.Module import MemAttention
from src.Module import MemEncoder
from src.Module import ImageDecoder, ImageDecoder_Start, ImageDecoder_End, ImageEncoder_MotObj
from sam2.build_sam import build_sam2_video_predictor, build_sam2

def export_image_encoder(model, onnx_path, dynamic_batch=True):
    input_img = torch.randn(1, 3, 1024, 1024).cpu()
    batch_size = torch.arange(2, dtype=torch.int32)  # shape [N], 与 C++ batch_size_info.size()=N 对齐

    for name, param in model.named_parameters(): 
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(image = input_img, batch_size = batch_size)

    output_names = ["pix_feat","high_res_feat0","high_res_feat1","vision_feats","vision_pos_embed"]

    if dynamic_batch:
        dyn_axes = {"batch_size": {0: "batch_size"}}
        torch.onnx.export(
            model,
            (input_img, batch_size),
            onnx_path+"image_encoder.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image", "batch_size"],
            output_names=output_names,
            dynamic_axes=dyn_axes,
        )

    else:
        torch.onnx.export(
            model,
            (input_img),
            onnx_path+"image_encoder.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["image"],
            output_names=output_names,
            # dynamic_axes=dyn_axes,
        )
    # 简化模型
    original_model = onnx.load(onnx_path+"image_encoder.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True, dynamic_input_shape=False)
    onnx.save(simplified_model, onnx_path+"image_encoder.onnx")
    #检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_encoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_encoder.onnx model is valid!")

def export_memory_attention(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(4,1,256)
    memory_1 = torch.randn(4,1,64,64,64)
    memory_pos_embed = torch.randn(4*4096+4*4,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention4_4.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention4_4.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention4_4.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention4_4.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")






def export_memory_attention5_5(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(5,1,256)
    memory_1 = torch.randn(5,1,64,64,64)
    memory_pos_embed = torch.randn(5*4096+5*4,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention5_5.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention5_5.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention5_5.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention5_5.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")



def export_memory_attention6_6(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(4,1,256)
    memory_1 = torch.randn(4,1,64,64,64)
    memory_pos_embed = torch.randn(4*4096+4*4,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention4_4.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention4_4.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention4_4.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention2_4.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_7(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(7,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*7,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_7.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_7.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_7.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_7.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_8(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(8,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*8,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_8.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_8.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_8.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_8.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_9(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(9,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*9,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_9.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_9.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_9.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_9.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_10(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(10,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*10,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_10.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_10.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_10.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_10.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_11(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(11,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*11,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_11.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_11.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_11.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_11.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_12(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(12,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*12,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_12.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_12.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_12.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_12.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_13(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(13,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*13,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_13.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_13.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_13.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_13.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_14(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(14,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*14,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性
 
    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_14.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_14.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_14.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_14.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")


def export_memory_attention7_15(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(15,1,256)
    memory_1 = torch.randn(7,1,64,64,64)
    memory_pos_embed = torch.randn(7*4096+4*15,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = {
        "current_vision_feat":{0: "batch_size_attention1"},
        "current_vision_pos_embed":{0: "batch_size_attention2", 1:"batch_size_attention6"},
        "memory_0": {0: "num",1:"batch_size_attention3"},
        "memory_1": {0: "buff_size",1:"batch_size_attention4"},
        "memory_pos_embed": {0: "buff_sizeb",1:"batch_size_attention5"},
    }
    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_15.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_15.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_15.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_15.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_memory_attention7_16(model, onnx_path, dynamic_batch=True):
    current_vision_feat = torch.randn(1,256,64,64)        #[1, 256, 64, 64],当前帧的视觉特征
    current_vision_pos_embed = torch.randn(4096,1,256)    #[4096, 1, 256],当前帧的位置特征
    memory_0 = torch.randn(16,1,256)
    memory_1 = torch.randn(7,1,64,64,64)

    memory_pos_embed = torch.randn(7*4096+4*16,1,64)      #[y*4096,1,64], 最近y帧的位置编码特性

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
            current_vision_feat = current_vision_feat,
            current_vision_pos_embed = current_vision_pos_embed,
            memory_0 = memory_0,
            memory_1 = memory_1,
            memory_pos_embed = memory_pos_embed,
        )

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_name = ["current_vision_feat",
                "current_vision_pos_embed",
                "memory_0",
                "memory_1",
                "memory_pos_embed"]
    dynamic_axes = None

    if dynamic_batch:
        dynamic_axes = {
            "current_vision_feat":{0: "batch_size_attention1"},
            "current_vision_pos_embed":{1:"batch_size_attention6"},
            "memory_0": {1:"batch_size_attention3"},
            "memory_1": {1:"batch_size_attention4"},
            "memory_pos_embed": {1:"batch_size_attention5"},
        }

    torch.onnx.export(
        model,
        (current_vision_feat,current_vision_pos_embed,memory_0,memory_1,memory_pos_embed),
        onnx_path+"memory_attention7_16.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=["image_embed"],
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
     # 简化模型,
    original_model = onnx.load(onnx_path+"memory_attention7_16.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_attention7_16.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_attention7_16.onnx")
    onnx.checker.check_model(onnx_model)
    print("memory_attention.onnx model is valid!")

def export_image_decoderStart(model, onnx_path, dynamic_batch=True):
    point_coords = torch.randint(0, 1024, (1, 2, 2), dtype=torch.float)

    point_labels = torch.tensor([[-1, -1]],dtype=torch.int32)
    image_embed = torch.randn(1,256,64,64).cpu()
    high_res_feats_0 = torch.randn(1,32,256,256).cpu()
    high_res_feats_1 = torch.randn(1,64,128,128).cpu()

    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        image_embed = image_embed,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    input_name = ["point_coords","point_labels","image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["object_score_logits","low_res_multimasks","high_res_multimasks", "sam_output_tokens", "ious", "low_res_masks", "high_res_masks", "sam_output_token"]
    dynamic_axes = {
        "point_coords":{0: "num_labels",1:"num_points"},
        "point_labels": {0: "num_labels_",1:"num_points_"},
        "image_embed": {0: "batch_size_start"},
        "high_res_feats_0": {0: "batch_size_start"},
        "high_res_feats_1": {0: "batch_size_start"},
    }
    torch.onnx.export(
        model,
        (point_coords,point_labels,image_embed,high_res_feats_0,high_res_feats_1),
        onnx_path+"image_decoderStart.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes if dynamic_batch else None,
    )
    # 简化模型,
    original_model = onnx.load(onnx_path+"image_decoderStart.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"image_decoderStart.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoderStart.onnx")
    onnx.checker.check_model(onnx_model)
    for input_node in onnx_model.graph.input:
        print(f"  Name: {input_node.name}, Shape: {input_node.type.tensor_type.shape}")
    print("image_decoderStart.onnx model is valid!")

def export_image_decoderEnd(model, onnx_path, dynamic_batch=True):
    object_score_logits = torch.randn(1,1).cpu()
    frame_size = torch.tensor([1080, 1920],dtype=torch.int32)
    low_res_masks = torch.randn(1,1,256,256).cpu()
    high_res_masks = torch.randn(1,1,1024,1024).cpu()
    sam_output_token = torch.randn(1,256).cpu()

    # for name, param in model.named_parameters():
    #     if param.dtype == torch.int64:
    #         param.data = param.data.to(torch.int32)


    out = model(
        object_score_logits = object_score_logits,
        frame_size = frame_size,
        low_res_masks = low_res_masks,
        high_res_masks = high_res_masks,
        sam_output_token = sam_output_token,
    )
    input_name = ["object_score_logits", "frame_size", "low_res_masks", "high_res_masks", "sam_output_token"]
    output_name = ["obj_ptr", "mask_for_mem", "pred_mask"]

    dynamic_axes = {
        "low_res_masks": {0: "batch_sizeEnd"},  # 动态第一个维度
        "high_res_masks": {0: "batch_sizeEnd"},  # 动态第一个维度
        "sam_output_token": {0: "batch_sizeEnd"},  # 动态第一个维度
        "frame_size": {0: "batch_size"},
        "object_score_logits": {0: "batch_sizeEnd"}  # 动态第一个维度
    }

    torch.onnx.export(
        model,
        (object_score_logits, frame_size, low_res_masks, high_res_masks, sam_output_token),
        onnx_path+"image_decoderEnd.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes
    )
    # 简化模型,
    original_model = onnx.load(onnx_path+"image_decoderEnd.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"image_decoderEnd.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoderEnd.onnx")
    onnx.checker.check_model(onnx_model)
    for input_node in onnx_model.graph.input:
        print(f"  Name: {input_node.name}, Shape: {input_node.type.tensor_type.shape}")
    print("image_decoderEnd.onnx model is valid!")
    for _ in range(10):
        print()

def export_image_decoder(model, onnx_path, dynamic_batch=True):
    point_coords = torch.randint(0, 1024, (1, 2, 2), dtype=torch.float)
    point_labels = torch.tensor([[-1, -1]],dtype=torch.int32)

    # frame_size = torch.tensor([1024,1024],dtype=torch.int32)
    image_embed = torch.randn(1,256,64,64).cpu()
    high_res_feats_0 = torch.randn(1,32,256,256).cpu()
    high_res_feats_1 = torch.randn(1,64,128,128).cpu()

    out = model(
        point_coords = point_coords,
        point_labels = point_labels,
        # frame_size = frame_size,
        image_embed = image_embed,
        high_res_feats_0 = high_res_feats_0,
        high_res_feats_1 = high_res_feats_1
    )
    input_name = ["point_coords","point_labels","image_embed","high_res_feats_0","high_res_feats_1"]
    output_name = ["obj_ptr","mask_for_mem","pred_mask","ious"]
    dynamic_axes = {
        "point_coords":{0: "num_labels",1:"num_points"},
        "point_labels": {0: "num_labels",1:"num_points"}
    }
    if dynamic_batch:
        dynamic_axes = {
            "point_coords":{0: "num_labels",1:"num_points"},
            "point_labels": {0: "num_labels",1:"num_points"},
            "image_embed": {0:"batch_size_start1"},
            "high_res_feats_0": {0:"batch_size_start2"},
            "high_res_feats_1": {0:"batch_size_start3"},
        }

    torch.onnx.export(
        model,
        (point_coords,point_labels,image_embed,high_res_feats_0,high_res_feats_1),
        onnx_path+"image_decoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names= input_name,
        output_names=output_name,
        dynamic_axes = dynamic_axes if dynamic_batch else None
    )
    # 简化模型,
    original_model = onnx.load(onnx_path+"image_decoder.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"image_decoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"image_decoder.onnx")
    onnx.checker.check_model(onnx_model)
    print("image_decoder.onnx model is valid!")

def export_memory_encoder(model, onnx_path, dynamic_batch=True):
    mask_for_mem = torch.randn(1,1,1024,1024)    #[batch_size, 1, 1024, 1024]
    pix_feat = torch.randn(1,256,64,64)

    #out = model(mask_for_mem = mask_for_mem,pix_feat = pix_feat)
    for name, param in model.named_parameters():
        if param.dtype == torch.int64:
            param.data = param.data.to(torch.int32)

    input_names = ["mask_for_mem","pix_feat"]
    output_names = ["maskmem_features","maskmem_pos_enc","temporal_code"]

    torch.onnx.export(
        model,
        (mask_for_mem, pix_feat),
        onnx_path+"memory_encoder.onnx",
        export_params=True,
        opset_version=17,
        do_constant_folding = True,
        input_names= input_names,
        output_names= output_names,
        dynamic_axes = {
            "mask_for_mem":{0: "batch_size_memencoder_p1"},
            "pix_feat":{0: "batch_size_memencoder_p1"}
        } if dynamic_batch else None
    )
    # 简化模型,
    original_model = onnx.load(onnx_path+"memory_encoder.onnx")
    if dynamic_batch:
        simplified_model, check = simplify(original_model)
    else:
        simplified_model, check = simplify(original_model, skip_shape_inference=True)
    onnx.save(simplified_model, onnx_path+"memory_encoder.onnx")
    # 检查检查.onnx格式是否正确
    onnx_model = onnx.load(onnx_path+"memory_encoder.onnx")
    for input_node in onnx_model.graph.input:
        print(f"  Name: {input_node.name}, Shape: {input_node.type.tensor_type.shape}")

    onnx.checker.check_model(onnx_model)
    print("memory_encoder.onnx model is valid!")

#****************************************************************************
model_type = ["tiny","small","large","base_plus"][1]
onnx_output_path = "checkpoints/small_mutilObj_nostart_end/"
model_config_file = "sam2_hiera_{}.yaml".format(model_type)
model_checkpoints_file = "checkpoints/sam2_hiera_{}.pt".format(model_type)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="导出SAM2为onnx文件")
    parser.add_argument("--outdir",type=str,default=onnx_output_path,required=False,help="path")
    parser.add_argument("--config",type=str,default=model_config_file,required=False,help="*.yaml")
    parser.add_argument("--checkpoint",type=str,default=model_checkpoints_file, required=False,help="*.pt")
    parser.add_argument("--static_batch", action="store_true", default=False,
                        help="导出静态shape ONNX（默认动态batch，支持多目标）")
    args = parser.parse_args()

    dynamic_batch = not args.static_batch


    sam2_model = build_sam2(args.config, args.checkpoint, device="cpu")


    # image_encoder = ImageEncoder(sam2_model).cpu()
    image_encoder = ImageEncoder_MotObj(sam2_model).cpu()
    export_image_encoder(image_encoder, args.outdir, dynamic_batch)



    # mem_attention = MemAttention(sam2_model).cpu() 
    # export_memory_attention7_16(mem_attention, args.outdir, dynamic_batch)


    # image_decoderStart = ImageDecoder_Start(sam2_model).cpu()
    # export_image_decoderStart(image_decoderStart,args.outdir, dynamic_batch)


    # image_decoderEnd = ImageDecoder_End(sam2_model).cpu()
    # export_image_decoderEnd(image_decoderEnd, args.outdir, dynamic_batch)


    # image_decoder   = ImageDecoder(sam2_model).cpu()
    # export_image_decoder(image_decoder, args.outdir, dynamic_batch)

    # mem_encoder   = MemEncoder(sam2_model).cpu()
    # export_memory_encoder(mem_encoder,args.outdir, dynamic_batch)











    # image_decoder = ImageDecoder(sam2_model).cpu()
    # export_image_decoder(image_decoder,args.outdir, dynamic_batch)
