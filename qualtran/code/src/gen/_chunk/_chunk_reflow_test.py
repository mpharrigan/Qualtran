import gen


def test_from_auto_rewrite_xs():
    result = gen.ChunkReflow.from_auto_rewrite(
        inputs=[
            gen.PauliMap({"X": [2, 3]}),
            gen.PauliMap({"X": [3, 4]}),
            gen.PauliMap({"X": [4, 5, 6]}),
            gen.PauliMap({"X": [5, 7]}),
            gen.PauliMap({"X": [8, 6]}),
            gen.PauliMap({"X": [7, 6]}),
        ],
        out2in={
            gen.PauliMap({"X": [2, 3]}): [gen.PauliMap({"X": [2, 3]})],
            gen.PauliMap({"X": [2]}): "auto",
        },
    )
    assert result == gen.ChunkReflow(
        out2in={
            gen.PauliMap({"X": [2, 3]}): [gen.PauliMap({"X": [2, 3]})],
            gen.PauliMap({"X": [2]}): [
                gen.PauliMap({"X": [2, 3]}),
                gen.PauliMap({"X": [3, 4]}),
                gen.PauliMap({"X": [4, 5, 6]}),
                gen.PauliMap({"X": [5, 7]}),
                gen.PauliMap({"X": [7, 6]}),
            ],
        }
    )


def test_from_auto_rewrite_xyz():
    result = gen.ChunkReflow.from_auto_rewrite(
        inputs=[gen.PauliMap({"X": [2, 3]}), gen.PauliMap({"Z": [2, 3]})],
        out2in={gen.PauliMap({"Y": [2, 3]}): "auto"},
    )
    assert result == gen.ChunkReflow(
        out2in={
            gen.PauliMap({"Y": [2, 3]}): [gen.PauliMap({"X": [2, 3]}), gen.PauliMap({"Z": [2, 3]})]
        }
    )


def test_from_auto_rewrite_keyed():
    result = gen.ChunkReflow.from_auto_rewrite(
        inputs=[gen.PauliMap({"X": [2, 3]}), gen.PauliMap({"Z": [2, 3]}).with_name("test")],
        out2in={gen.PauliMap({"Y": [2, 3]}): "auto"},
    )
    assert result == gen.ChunkReflow(
        out2in={
            gen.PauliMap({"Y": [2, 3]}): [
                gen.PauliMap({"X": [2, 3]}),
                gen.PauliMap({"Z": [2, 3]}).with_name("test"),
            ]
        }
    )


def test_from_auto_rewrite_transitions_using_stable():
    x12 = gen.PauliMap.from_xs([1, 2])
    y12 = gen.PauliMap.from_ys([1, 2])
    z12 = gen.PauliMap.from_zs([1, 2])
    x1 = gen.PauliMap.from_xs([1])
    x2 = gen.PauliMap.from_xs([2])
    assert gen.ChunkReflow.from_auto_rewrite_transitions_using_stable(
        stable=[x12], transitions=[(x1, x2)]
    ) == gen.ChunkReflow(out2in={x12: [x12], x2: [x12, x1]})
    assert gen.ChunkReflow.from_auto_rewrite_transitions_using_stable(
        stable=[y12], transitions=[(z12.with_name("test"), x12.with_name("test"))]
    ) == gen.ChunkReflow(out2in={y12: [y12], x12.with_name("test"): [y12, z12.with_name("test")]})
