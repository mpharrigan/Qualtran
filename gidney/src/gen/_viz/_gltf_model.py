from __future__ import annotations

import io
import pathlib

import pygltflib


class gltf_model(pygltflib.GLTF2):
    """A pygltflib.GLTF2 augmented with _repr_html_ and a `write_to` method."""

    def _repr_html_(self) -> str:
        """This is the method Jupyter notebooks look for, to show as an SVG."""
        from gen._viz._viz_gltf_3d import viz_3d_gltf_model_html

        return viz_3d_gltf_model_html(self)

    def write_viewer_to(self, path: str | pathlib.Path | io.IOBase):
        from gen._viz._viz_gltf_3d import viz_3d_gltf_model_html

        viz_3d_gltf_model_html(self).write_to(path)
