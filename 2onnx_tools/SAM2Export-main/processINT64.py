import onnx
from onnx import TensorProto
from onnx import helper

# 加载 ONNX 模型
onnx_model = onnx.load("checkpoints/base_plus/memory_attention4_4.onnx")



# 假设模型已经加载为 onnx_model
# 找到 /Unsqueeze_1_output_0 的节点
# 在其后面插入一个 Cast 节点，将其输出类型转换为 INT64

cast_node1 = helper.make_node(
    'Cast',
    inputs=['/Unsqueeze_1_output_0'],
    outputs=['/Unsqueeze_1_output_0_cast'],
    to=onnx.TensorProto.INT64
)

# 将 Cast 节点插入到模型中
onnx_model.graph.node.append(cast_node1)

# 更新 /Concat_1 的输入，使用 Cast 后的输出
for node in onnx_model.graph.node:
    if node.name == '/Concat_1':
        node.input[1] = '/Unsqueeze_1_output_0_cast'



cast_node2 = helper.make_node(
    'Cast',
    inputs=['/Unsqueeze_2_output_0'],
    outputs=['/Unsqueeze_2_output_0_cast'],
    to=onnx.TensorProto.INT64
)

# 将 Cast 节点插入到模型中
onnx_model.graph.node.append(cast_node2)

# 更新 /Concat_4 的输入，使用 Cast 后的输出
for node in onnx_model.graph.node:
    if node.name == '/Concat_4':
        node.input[1] = '/Unsqueeze_2_output_0_cast'

# 保存修改后的模型
onnx.save(onnx_model, 'memory_attention_modified_final4_4.onnx')
