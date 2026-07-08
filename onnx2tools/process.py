import onnx
import onnx_graphsurgeon as gs
import numpy as np

INPUT = "checkpoints/small_mutilObj_nostart_end/memory_attention7_16.onnx"
OUTPUT = "checkpoints/small_mutilObj_nostart_end/memory_attention7_16_fix.onnx"

graph = gs.import_onnx(onnx.load(INPUT))

new_nodes = []

for node in graph.nodes:

    if node.op != "Concat":
        continue

    for i, inp in enumerate(node.inputs):

        # 只处理来自 Unsqueeze 的 Shape Tensor
        producer = inp.inputs

        if len(producer) == 0:
            continue

        if producer[0].op != "Unsqueeze":
            continue

        cast_out = gs.Variable(
            name=inp.name + "_cast_int64",
            dtype=np.int64,
        )

        cast = gs.Node(
            op="Cast",
            name=inp.name + "_CastToINT64",
            attrs={"to": onnx.TensorProto.INT64},
            inputs=[inp],
            outputs=[cast_out],
        )

        new_nodes.append(cast)

        node.inputs[i] = cast_out

        print(f"Patch {node.name} input {i}")

graph.nodes.extend(new_nodes)

graph.cleanup()
graph.toposort()

onnx.save(gs.export_onnx(graph), OUTPUT)

print("Saved:", OUTPUT)