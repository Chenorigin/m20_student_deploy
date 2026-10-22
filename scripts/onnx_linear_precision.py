"""Validate a feedforward ELU MLP and select its linear compute precision."""
import copy
import hashlib
import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def _shape(value):
    tensor = value.type.tensor_type
    dims = tensor.shape.dim
    if (tensor.elem_type != TensorProto.FLOAT or len(dims) != 2
            or any(not dim.HasField("dim_value") or dim.dim_value <= 0 for dim in dims)
            or dims[0].dim_value != 1):
        raise ValueError("MLP I/O must be static float32 [1,features]")
    return [dim.dim_value for dim in dims]


def _mlp_geometry(model):
    graph = model.graph
    if len(graph.input) != 1 or len(graph.output) != 1 or graph.sparse_initializer:
        raise ValueError("Expected a single-input/single-output dense MLP")
    input_shape, output_shape = _shape(graph.input[0]), _shape(graph.output[0])
    nodes = list(graph.node)
    if not nodes or len(nodes) % 2 != 1:
        raise ValueError("Expected alternating Gemm/ELU nodes ending in Gemm")
    initializers = {item.name: item for item in graph.initializer}
    if len(initializers) != len(graph.initializer):
        raise ValueError("Duplicate initializer name")
    values = {}
    for name, item in initializers.items():
        value = numpy_helper.to_array(item)
        if value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("MLP weights and biases must be finite float32: " + name)
        values[name] = value
    current, width = graph.input[0].name, input_shape[1]
    used, produced, widths = set(), {current, *initializers}, []
    for index, node in enumerate(nodes):
        expected = "Gemm" if index % 2 == 0 else "Elu"
        if node.domain not in ("", "ai.onnx") or node.op_type != expected:
            raise ValueError("Unsupported MLP graph: expected " + expected)
        if (len(node.output) != 1 or not node.output[0] or node.output[0] in produced
                or not node.input or node.input[0] != current):
            raise ValueError("MLP must be an unbranched chain with unique tensor names")
        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        if len(attrs) != len(node.attribute):
            raise ValueError("Duplicate node attribute")
        if expected == "Gemm":
            if (len(node.input) != 3 or set(attrs) - {"alpha", "beta", "transA", "transB"}
                    or attrs.get("alpha", 1.) != 1. or attrs.get("beta", 1.) != 1.
                    or attrs.get("transA", 0) != 0 or attrs.get("transB", 0) not in (0, 1)
                    or any(name not in values for name in node.input[1:])):
                raise ValueError("Unsupported MLP Gemm or nonconstant parameters")
            weight, bias = (values[name] for name in node.input[1:])
            if weight.ndim != 2 or bias.ndim != 1:
                raise ValueError("Expected matrix weight and vector bias")
            left, right = weight.shape[::-1] if attrs.get("transB", 0) else weight.shape
            if left != width or right <= 0 or bias.shape != (right,):
                raise ValueError("MLP weight/bias dimensions do not form a chain")
            width = right
            widths.append(width)
            used.update(node.input[1:])
        elif len(node.input) != 1 or set(attrs) - {"alpha"} or attrs.get("alpha", 1.) != 1.:
            raise ValueError("Only ELU alpha=1 is supported")
        current = node.output[0]
        produced.add(current)
    if current != graph.output[0].name or width != output_shape[1] or used != set(initializers):
        raise ValueError("Unexpected MLP output, unused parameters or output dimensions")
    onnx.checker.check_model(model)
    return {"input_shape": input_shape, "output_shape": output_shape,
            "hidden_dims": widths[:-1], "linear_layers": len(widths)}


def set_linear_precision(model, linear_precision="float64"):
    """Return a new validated model, precision metadata and inferred MLP sizes.

    FP64 exactly represents every finite FP32 parameter. Each linear result is
    rounded back to FP32 before the original ELU; external I/O remains FP32.
    The float32 option keeps the original arithmetic for numerical comparisons.
    """
    if linear_precision not in ("float32", "float64"):
        raise ValueError("Unsupported linear precision")
    geometry = _mlp_geometry(model)
    if any(prop.key == "precision" for prop in model.metadata_props):
        raise ValueError("Source model already declares linear precision")
    source_hash = hashlib.sha256(model.SerializeToString()).hexdigest()
    result = copy.deepcopy(model)
    if linear_precision == "float64":
        for item in result.graph.initializer:
            original = numpy_helper.to_array(item)
            widened = original.astype(np.float64)
            if not np.array_equal(widened.astype(np.float32).view(np.uint32), original.view(np.uint32)):
                raise ValueError("FP32 parameter widening lost an original value")
            item.CopyFrom(numpy_helper.from_array(widened, item.name))
        used = {value.name for value in (*result.graph.input, *result.graph.initializer)}
        used.update(output for node in result.graph.node for output in node.output)

        def unique(base):
            name, suffix = base, 0
            while name in used:
                suffix += 1
                name = base + "_" + str(suffix)
            used.add(name)
            return name

        nodes = []
        for index, node in enumerate(result.graph.node):
            if node.op_type != "Gemm":
                nodes.append(copy.deepcopy(node))
                continue
            widened = copy.deepcopy(node)
            widened.input[0] = unique(node.input[0] + "__linear_fp64")
            widened.output[0] = unique(node.output[0] + "__linear_fp64")
            nodes.extend((
                helper.make_node("Cast", [node.input[0]], [widened.input[0]],
                                 name="linear_precision_input_" + str(index), to=TensorProto.DOUBLE),
                widened,
                helper.make_node("Cast", [widened.output[0]], [node.output[0]],
                                 name="linear_precision_output_" + str(index), to=TensorProto.FLOAT)))
        del result.graph.node[:]
        result.graph.node.extend(nodes)
    precision = {
        "version": 1,
        "linear_compute_dtype": linear_precision,
        "parameter_source_dtype": "float32",
        "parameter_storage_dtype": linear_precision,
        "parameter_values": "original float32 values preserved exactly",
        "linear_output_dtype": "float32",
        "activation": "ELU alpha=1 in float32",
        "external_input_dtype": "float32",
        "external_output_dtype": "float32",
        "rounding": "float32 after every linear layer, before activation",
        "source_onnx_sha256": source_hash,
    }
    # Append rather than rebuilding metadata so every existing entry is retained.
    entry = result.metadata_props.add()
    entry.key = "precision"
    entry.value = json.dumps(precision, sort_keys=True, separators=(",", ":"))
    onnx.checker.check_model(result)
    onnx.shape_inference.infer_shapes(result, check_type=True, strict_mode=True)
    return result, precision, geometry
