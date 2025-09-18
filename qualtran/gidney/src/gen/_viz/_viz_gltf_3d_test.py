import json

import numpy as np
import pytest

import gen
from gen._viz._viz_gltf_3d import TextData


def test_gltf_model_from_colored_triangle_data():
    pytest.importorskip("pygltflib")

    model = gen.gltf_model_from_colored_triangle_data(
        [
            gen.ColoredTriangleData(
                rgba=(1, 0, 0, 1),
                triangle_list=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            ),
            gen.ColoredTriangleData(
                rgba=(1, 0, 1, 1),
                triangle_list=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            ),
        ]
    )
    assert json.loads(model.to_json()) == {
        "accessors": [
            {
                "bufferView": 0,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "max": [1.0, 1.0, 1.0],
                "min": [0.0, 0.0, 0.0],
                "normalized": False,
                "type": "VEC3",
            },
            {
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5126,
                "count": 3,
                "max": [1.0, 1.0, 1.0],
                "min": [0.0, 0.0, 0.0],
                "normalized": False,
                "type": "VEC3",
            },
        ],
        "asset": {"version": "2.0"},
        "bufferViews": [
            {"buffer": 0, "byteLength": 36, "byteOffset": 0, "target": 34962},
            {"buffer": 0, "byteLength": 36, "byteOffset": 36, "target": 34962},
        ],
        "buffers": [
            {
                "byteLength": 72,
                "uri": "data:application/octet-stream;base64,AACAPwAAAAAAAAAAAAAAAAAAgD8AAAAAAAAA"
                "AAAAAAAAAIA/AACAPwAAgD8AAAAAAAAAAAAAgD8AAAAAAAAAAAAAAAAAAIA/",
            }
        ],
        "materials": [
            {
                "alphaCutoff": 0.5,
                "alphaMode": "OPAQUE",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 0, 0, 1],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.8,
                },
            },
            {
                "alphaCutoff": 0.5,
                "alphaMode": "OPAQUE",
                "doubleSided": True,
                "pbrMetallicRoughness": {
                    "baseColorFactor": [1, 0, 1, 1],
                    "metallicFactor": 0.3,
                    "roughnessFactor": 0.8,
                },
            },
        ],
        "meshes": [
            {
                "primitives": [
                    {"attributes": {"POSITION": 0, "TEXCOORD_0": 0}, "material": 0, "mode": 4},
                    {"attributes": {"POSITION": 1, "TEXCOORD_0": 1}, "material": 1, "mode": 4},
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
    }


def test_viz_3d_gltf_model_html():
    pytest.importorskip("pygltflib")

    model = gen.gltf_model_from_colored_triangle_data(
        [
            gen.ColoredTriangleData(
                rgba=(1, 0, 0, 1),
                triangle_list=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            ),
            gen.ColoredTriangleData(
                rgba=(1, 0, 1, 1),
                triangle_list=np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32),
            ),
        ]
    )

    html = gen.viz_3d_gltf_model_html(model)
    assert "<html>" in html


def test_3d_text():
    model = gen.gltf_model_from_colored_triangle_data(
        [
            gen.ColoredTriangleData(
                rgba=(1, 0, 0, 1),
                triangle_list=np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32),
            )
        ],
        colored_line_data=[
            gen.ColoredLineData(
                rgba=(0, 0, 1, 1), edge_list=np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
            )
        ],
        text_data=[TextData(text="test", start=[0, 0, 0], forward=(1, 0, 0), up=(0, 1, 0))],
    )
    assert model is not None
