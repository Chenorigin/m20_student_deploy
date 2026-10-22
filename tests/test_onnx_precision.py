"""CPU export checks for exact parameter widening and the unchanged parity gate."""
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from onnx_linear_precision import set_linear_precision


def actor(hidden):
    torch.set_num_threads(1)
    torch.manual_seed(42019)
    modules = []
    sizes = [789, *hidden, 16]
    for index, (left, right) in enumerate(zip(sizes, sizes[1:])):
        modules.append(torch.nn.Linear(left, right))
        if index < len(sizes) - 2:
            modules.append(torch.nn.ELU())
    result = torch.nn.Sequential(*modules).eval()
    with torch.no_grad():
        result[0].weight[0, 0] = -0.
    return result


def raw_graph(model):
    buffer = io.BytesIO()
    torch.onnx.export(torch.jit.script(model), torch.zeros(1, 789), buffer,
                      input_names=["obs"], output_names=["actions"],
                      opset_version=17, dynamo=False, do_constant_folding=True)
    return onnx.load_model_from_string(buffer.getvalue())


def assert_transformation(source, result, precision):
    source_values = {value.name: numpy_helper.to_array(value) for value in source.graph.initializer}
    assert {value.name for value in result.graph.initializer} == set(source_values)
    for item in result.graph.initializer:
        widened = numpy_helper.to_array(item)
        assert widened.dtype == (np.float64 if precision == "float64" else np.float32)
        original = source_values[item.name]
        np.testing.assert_array_equal(widened, original.astype(widened.dtype))
        np.testing.assert_array_equal(widened.astype(np.float32).view(np.uint32), original.view(np.uint32))
    for old, new in zip(source.graph.input, result.graph.input):
        assert old.SerializeToString() == new.SerializeToString()
        assert new.type.tensor_type.elem_type == TensorProto.FLOAT
    for old, new in zip(source.graph.output, result.graph.output):
        assert old.SerializeToString() == new.SerializeToString()
        assert new.type.tensor_type.elem_type == TensorProto.FLOAT
    old_metadata = [entry.SerializeToString() for entry in source.metadata_props]
    assert [entry.SerializeToString() for entry in result.metadata_props[:-1]] == old_metadata
    assert result.metadata_props[-1].key == "precision"
    assert json.loads(result.metadata_props[-1].value)["linear_compute_dtype"] == precision
    nodes = iter(result.graph.node)
    for original in source.graph.node:
        if precision == "float64" and original.op_type == "Gemm":
            before, linear, after = next(nodes), next(nodes), next(nodes)
            assert [before.op_type, linear.op_type, after.op_type] == ["Cast", "Gemm", "Cast"]
            assert helper.get_attribute_value(before.attribute[0]) == TensorProto.DOUBLE
            assert helper.get_attribute_value(after.attribute[0]) == TensorProto.FLOAT
            assert list(before.input) == [original.input[0]]
            assert list(linear.input) == [before.output[0], *original.input[1:]]
            assert list(after.input) == list(linear.output)
            assert list(after.output) == list(original.output)
            assert [attr.SerializeToString() for attr in linear.attribute] == [
                attr.SerializeToString() for attr in original.attribute]
        else:
            assert next(nodes).SerializeToString() == original.SerializeToString()
    assert next(nodes, None) is None
    onnx.checker.check_model(result)


@pytest.mark.parametrize("hidden", [[9], [31, 19], [23, 17, 11, 5]])
def test_arbitrary_mlp_widths_preserve_values_topology_and_metadata(hidden):
    source = raw_graph(actor(hidden))
    helper.set_model_props(source, {"hidden_dims": json.dumps(hidden), "opaque": "retain\nexactly"})
    before = source.SerializeToString()
    result, precision, geometry = set_linear_precision(source)
    assert source.SerializeToString() == before
    assert geometry["hidden_dims"] == hidden
    assert geometry["input_shape"] == [1, 789]
    assert geometry["output_shape"] == [1, 16]
    assert precision["source_onnx_sha256"] == hashlib.sha256(before).hexdigest()
    assert_transformation(source, result, "float64")


@pytest.mark.parametrize("case", ["relu", "alpha", "transA", "attribute", "branch",
                                  "missing_bias", "bias_shape", "unused", "nonfinite",
                                  "weight_dtype", "io_dtype", "dynamic", "duplicate_tensor",
                                  "output_width", "precision_metadata"])
def test_unsupported_graphs_fail_closed(case):
    source = raw_graph(actor([7, 5]))
    if case == "relu":
        source.graph.node[1].op_type = "Relu"
        del source.graph.node[1].attribute[:]
    elif case == "alpha":
        source.graph.node[1].attribute[0].f = .5
    elif case == "transA":
        source.graph.node[0].attribute.append(helper.make_attribute("transA", 1))
    elif case == "attribute":
        source.graph.node[0].attribute.append(helper.make_attribute("unknown", 1))
    elif case == "branch":
        source.graph.node[2].input[0] = "obs"
    elif case == "missing_bias":
        del source.graph.node[0].input[2]
    elif case == "bias_shape":
        item = source.graph.initializer[1]
        item.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(item)[:-1], item.name))
    elif case == "unused":
        source.graph.initializer.append(numpy_helper.from_array(np.ones(1, np.float32), "unused"))
    elif case in ("nonfinite", "weight_dtype"):
        item = source.graph.initializer[0]
        array = numpy_helper.to_array(item).copy()
        if case == "nonfinite":
            array[0, 0] = np.nan
        else:
            array = array.astype(np.float64)
        item.CopyFrom(numpy_helper.from_array(array, item.name))
    elif case == "io_dtype":
        source.graph.input[0].type.tensor_type.elem_type = TensorProto.DOUBLE
    elif case == "dynamic":
        dim = source.graph.input[0].type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "batch"
    elif case == "duplicate_tensor":
        source.graph.node[1].output[0] = source.graph.node[0].output[0]
    elif case == "output_width":
        source.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 17
    elif case == "precision_metadata":
        helper.set_model_props(source, {"precision": "{}"})
    with pytest.raises(ValueError):
        set_linear_precision(source)


def export(model_path, output, *extra, succeeds=True):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/export_onnx.py"),
                             "--model", str(model_path), "--profile", "v1",
                             "--contract", str(ROOT / "models/v1/observation_contract.json"),
                             "--output", str(output), *extra],
                            capture_output=True, text=True, env=env, timeout=60)
    if succeeds:
        assert result.returncode == 0, result.stdout + result.stderr
        return json.loads(output.with_suffix(".onnx.json").read_text())
    assert result.returncode != 0
    return result


@pytest.mark.parametrize("precision", ["float64", "float32"])
def test_torchscript_cli_export_ort_and_precision_receipt(tmp_path, precision):
    source_actor = actor([19, 11])
    model_path = tmp_path / "source.pt"
    torch.jit.save(torch.jit.script(source_actor), str(model_path))
    output = tmp_path / "policy.onnx"
    options = () if precision == "float64" else ("--linear-precision", "float32")
    record = export(model_path, output, *options)
    assert record["linear_precision"] == precision
    assert record["hidden_dims"] == [19, 11]
    assert record["source_model_sha256"] == hashlib.sha256(model_path.read_bytes()).hexdigest()
    assert record["source_model_sha256"] == record["torchscript_sha256"]
    assert record["source_onnx_sha256"] == record["precision"]["source_onnx_sha256"]
    assert record["parity_samples"] == 32
    assert record["parity_atol"] == record["parity_rtol"] == 2e-5
    graph = onnx.load(str(output))
    raw = raw_graph(source_actor)
    raw.metadata_props.extend(graph.metadata_props[:-1])
    assert_transformation(raw, graph, precision)
    assert {value.name: numpy_helper.to_array(value).dtype for value in graph.graph.initializer} == {
        value.name: (np.dtype("float64") if precision == "float64" else np.dtype("float32"))
        for value in raw.graph.initializer}
    props = {item.key: item.value for item in graph.metadata_props}
    assert props["deployment_family"] == "m20_student_v3_5"
    assert json.loads(props["precision"]) == record["precision"]
    assert "candidate" not in props["precision"] and "status" not in props["precision"]
    settings = ort.SessionOptions()
    settings.intra_op_num_threads = settings.inter_op_num_threads = 1
    settings.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(str(output), sess_options=settings, providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(42)
    for _ in range(32):
        x = rng.normal(size=(1, 789)).astype(np.float32)
        with torch.inference_mode():
            expected = source_actor(torch.from_numpy(x)).numpy()
        observed = session.run(None, {"obs": x})[0]
        np.testing.assert_allclose(observed, expected, atol=2e-5, rtol=2e-5)
    original = output.read_bytes()
    export(model_path, output, *options, succeeds=False)
    assert output.read_bytes() == original


def test_bad_torchscript_is_rejected_without_output(tmp_path):
    bad = torch.nn.Sequential(torch.nn.Linear(789, 7), torch.nn.ReLU(), torch.nn.Linear(7, 16)).eval()
    source = tmp_path / "bad.pt"
    torch.jit.save(torch.jit.script(bad), str(source))
    output = tmp_path / "bad.onnx"
    result = export(source, output, succeeds=False)
    assert "Unsupported MLP graph" in result.stderr
    assert not output.exists() and not output.with_suffix(".onnx.json").exists()


def test_current_release_export_all_frozen_inputs(tmp_path):
    model_path = ROOT / "models/v1/policy.pt"
    output = tmp_path / "policy.onnx"
    record = export(model_path, output)
    assert record["linear_precision"] == "float64"
    metadata = json.loads(model_path.with_name("deployment.json").read_text())
    assert record["hidden_dims"] == metadata["hidden_dims"]
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.add_session_config_entry("session.disable_prepacking", "1")
    session = ort.InferenceSession(str(output), sess_options=options, providers=["CPUExecutionProvider"])
    fixtures = ROOT / "evidence/offline_models/v1"
    x = np.fromfile(fixtures / "input.f32", dtype="<f4").reshape(-1, 789)
    expected = np.fromfile(fixtures / "expected.f32", dtype="<f4").reshape(-1, 16)
    assert len(x) == 160
    with np.load(ROOT / "evidence/numerics/real_v3_historical_capture.npz") as capture:
        datasets = [(x, expected), (capture["obs"].copy(), capture["expected"].copy())]
    assert len(datasets[1][0]) == 8587
    for inputs, reference in datasets:
        observed = np.concatenate([session.run(None, {"obs": row[None]})[0] for row in inputs])
        np.testing.assert_allclose(observed, reference, atol=2e-5, rtol=2e-5)
