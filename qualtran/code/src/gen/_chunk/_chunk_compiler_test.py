import stim

import gen


def test_chunk_compiler_q2i():
    compiler = gen.ChunkCompiler()
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(2, 3) 0
            H 0
        """
            ),
            flows=[],
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
                QUBIT_COORDS(2, 4) 0
                QUBIT_COORDS(2, 3) 1
                CX 0 1
                """
            ),
            flows=[],
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
                QUBIT_COORDS(2, 4) 0
                S 0
                """
            ),
            flows=[],
        )
    )
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(2, 4) 1
        H 0
        TICK
        CX 1 0
        TICK
        S 1
            """
    )


def test_chunk_compiler_single_flow():
    compiler = gen.ChunkCompiler()
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
                QUBIT_COORDS(1, 2) 0
                R 0
                """
            ),
            flows=[gen.Flow(end=gen.PauliMap.from_zs([1 + 2j]), center=3 + 5j)],
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
                QUBIT_COORDS(1, 2) 0
                M 0
                """
            ),
            flows=[gen.Flow(start=gen.PauliMap.from_zs([1 + 2j]), mids=[0], center=3 + 5j)],
        )
    )
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(1, 2) 0
        R 0
        TICK
        M 0
        DETECTOR(3, 5, 0) rec[-1]
            """
    )


def test_chunk_compiler_obs_flow_eager_dump():
    compiler = gen.ChunkCompiler()
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            R 0
        """
            ),
            flows=[gen.Flow(end=gen.PauliMap.from_zs([0]), center=0, obs_key=0)],
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            MR 0
        """
            ),
            flows=[
                gen.Flow(
                    start=gen.PauliMap.from_zs([0]),
                    end=gen.PauliMap.from_zs([0]),
                    mids=[0],
                    center=0,
                    obs_key=0,
                )
            ],
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            M 0
        """
            ),
            flows=[gen.Flow(start=gen.PauliMap.from_zs([0]), mids=[0], center=0, obs_key=0)],
        )
    )
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        MR 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        TICK
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )


def test_chunk_compiler_loop():
    compiler = gen.ChunkCompiler()
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(0, 1) 1
            QUBIT_COORDS(0, 2) 2
            QUBIT_COORDS(0, 3) 3
            R 0 1 2 3
        """
            ),
            flows=[gen.Flow(end=gen.PauliMap.from_zs([k]), center=0) for k in range(4)],
        )
    )
    compiler.append(
        gen.ChunkLoop(
            [
                gen.Chunk(
                    circuit=stim.Circuit(
                        """
                    QUBIT_COORDS(0, 0) 0
                    QUBIT_COORDS(0, 1) 1
                    QUBIT_COORDS(0, 2) 2
                    QUBIT_COORDS(0, 3) 3
                    SWAP 0 1
                    SWAP 1 2
                    SWAP 2 3
                    M 3
                """
                    ),
                    flows=[
                        gen.Flow(start=gen.PauliMap.from_zs([0]), mids=[0], center=0),
                        gen.Flow(end=gen.PauliMap.from_zs([3]), mids=[0], center=0),
                        gen.Flow(
                            start=gen.PauliMap.from_zs([1]), end=gen.PauliMap.from_zs([0]), center=0
                        ),
                        gen.Flow(
                            start=gen.PauliMap.from_zs([2]), end=gen.PauliMap.from_zs([1]), center=0
                        ),
                        gen.Flow(
                            start=gen.PauliMap.from_zs([3]), end=gen.PauliMap.from_zs([2]), center=0
                        ),
                    ],
                )
            ],
            repetitions=1000,
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(0, 1) 1
            QUBIT_COORDS(0, 2) 2
            QUBIT_COORDS(0, 3) 3
            M 0 1 2 3
        """
            ),
            flows=[gen.Flow(start=gen.PauliMap.from_zs([k]), mids=[k], center=0) for k in range(4)],
        )
    )
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(0, 2) 2
        QUBIT_COORDS(0, 3) 3
        R 0 1 2 3
        TICK
        REPEAT 4 {
            SWAP 0 1 1 2 2 3
            M 3
            DETECTOR(0, 0, 0) rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        REPEAT 996 {
            SWAP 0 1 1 2 2 3
            M 3
            DETECTOR(0, 0, 0) rec[-5] rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        M 0 1 2 3
        DETECTOR(0, 0, 0) rec[-8] rec[-4]
        DETECTOR(0, 0, 1) rec[-7] rec[-3]
        DETECTOR(0, 0, 2) rec[-6] rec[-2]
        DETECTOR(0, 0, 3) rec[-5] rec[-1]
        """
    )


def test_chunk_compiler_loop_obs():
    compiler = gen.ChunkCompiler()
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            R 0
        """
            ),
            flows=[gen.Flow(end=gen.PauliMap.from_zs([0]), center=0, obs_key=3)],
        )
    )
    compiler.append(
        gen.ChunkLoop(
            [
                gen.Chunk(
                    circuit=stim.Circuit(
                        """
                    QUBIT_COORDS(0, 0) 0
                    MR 0
                """
                    ),
                    flows=[
                        gen.Flow(
                            start=gen.PauliMap.from_zs([0]),
                            end=gen.PauliMap.from_zs([0]),
                            mids=[0],
                            center=0,
                            obs_key=3,
                        )
                    ],
                )
            ],
            repetitions=1000,
        )
    )
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 0) 0
            M 0
        """
            ),
            flows=[gen.Flow(start=gen.PauliMap.from_zs([0]), mids=[0], center=0, obs_key=3)],
        )
    )
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        REPEAT 1000 {
            MR 0
            OBSERVABLE_INCLUDE(0) rec[-1]
            TICK
        }
        M 0
        OBSERVABLE_INCLUDE(0) rec[-1]
        """
    )


def test_compile_postselected_chunks():
    chunk1 = gen.Chunk(
        circuit=stim.Circuit(
            """
            R 0
        """
        ),
        q2i={0: 0},
        flows=[gen.Flow(center=0, end=gen.PauliMap({0: "Z"}))],
    )
    chunk2 = gen.Chunk(
        circuit=stim.Circuit(
            """
            M 0
        """
        ),
        q2i={0: 0},
        flows=[
            gen.Flow(center=0, end=gen.PauliMap({0: "Z"}), mids=[0]),
            gen.Flow(center=0, start=gen.PauliMap({0: "Z"}), mids=[0]),
        ],
    )
    chunk3 = gen.Chunk(
        circuit=stim.Circuit(
            """
            MR 0
        """
        ),
        q2i={0: 0},
        flows=[gen.Flow(center=0, start=gen.PauliMap({0: "Z"}), mids=[0])],
    )

    assert gen.compile_chunks_into_circuit([chunk1, chunk2, chunk3]).flattened() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        M 0
        DETECTOR(0, 0, 0) rec[-1]
        TICK
        MR 0
        DETECTOR(0, 0, 1) rec[-2] rec[-1]
        """
    )

    assert gen.compile_chunks_into_circuit(
        [
            chunk1.with_edits(flows=[f.with_edits(flags={"postselect"}) for f in chunk1.flows]),
            chunk2,
            chunk3,
        ],
        metadata_func=lambda flow: gen.FlowMetadata(
            extra_coords=[999] if "postselect" in flow.flags else []
        ),
    ).flattened() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
            R 0
            TICK
            M 0
            DETECTOR(0, 0, 0, 999) rec[-1]
            TICK
            MR 0
            DETECTOR(0, 0, 1) rec[-2] rec[-1]
            """
    )

    assert gen.compile_chunks_into_circuit(
        [
            chunk1,
            chunk2.with_edits(flows=[f.with_edits(flags={"postselect"}) for f in chunk2.flows]),
            chunk3,
        ],
        metadata_func=lambda flow: gen.FlowMetadata(
            extra_coords=[999] if "postselect" in flow.flags else []
        ),
    ).flattened() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        M 0
        DETECTOR(0, 0, 0, 999) rec[-1]
        TICK
        MR 0
        DETECTOR(0, 0, 1, 999) rec[-2] rec[-1]
    """
    )

    assert gen.compile_chunks_into_circuit(
        [
            chunk1,
            chunk2,
            chunk3.with_edits(flows=[f.with_edits(flags={"postselect"}) for f in chunk3.flows]),
        ],
        metadata_func=lambda flow: gen.FlowMetadata(
            extra_coords=[999] if "postselect" in flow.flags else []
        ),
    ).flattened() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        M 0
        DETECTOR(0, 0, 0) rec[-1]
        TICK
        MR 0
        DETECTOR(0, 0, 1, 999) rec[-2] rec[-1]
    """
    )

    assert gen.compile_chunks_into_circuit(
        [
            chunk1,
            chunk2.with_edits(
                flows=[f.with_edits(flags={"postselect"}) if f.start else f for f in chunk2.flows]
            ),
            chunk3,
        ],
        metadata_func=lambda flow: gen.FlowMetadata(
            extra_coords=[999] if "postselect" in flow.flags else []
        ),
    ).flattened() == stim.Circuit(
        """
            QUBIT_COORDS(0, 0) 0
            R 0
            TICK
            M 0
            DETECTOR(0, 0, 0, 999) rec[-1]
            TICK
            MR 0
            DETECTOR(0, 0, 1) rec[-2] rec[-1]
        """
    )


def test_chunk_compiler_propagate_discards():
    c = gen.ChunkCompiler()
    xx = gen.PauliMap.from_xs([0, 1])
    zz = gen.PauliMap.from_zs([0, 1])
    c.append(
        gen.Chunk(
            stim.Circuit(
                """
                    R 0 1
                """
            ),
            q2i={0: 0, 1: 1},
            flows=[gen.Flow(end=zz, center=0)],
            discarded_outputs=[xx],
        )
    )
    c.append(
        gen.ChunkSemiAuto(
            stim.Circuit(
                """
                    MZZ 0 1
                """
            ),
            q2i={0: 0, 1: 1},
            flows=[
                gen.FlowSemiAuto(start=zz, center=0, mids="auto"),
                gen.FlowSemiAuto(end=zz, center=0, mids="auto"),
                gen.FlowSemiAuto(start=xx, end=xx, center=0),
            ],
        )
    )
    c.append(
        gen.ChunkSemiAuto(
            stim.Circuit(
                """
                    MX 0 1
                """
            ),
            q2i={0: 0, 1: 1},
            discarded_inputs=[zz],
            flows=[gen.FlowSemiAuto(start=xx, center=0, mids="auto")],
        )
    )
    assert c.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        R 0 1
        TICK
        MZZ 0 1
        DETECTOR(0, 0, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        MX 0 1
    """
    )


def test_drop_observable_later():
    c = gen.ChunkCompiler()
    xx = gen.PauliMap.from_xs([0, 1])
    zz = gen.PauliMap.from_zs([0, 1])
    c.append(
        gen.Chunk(
            stim.Circuit(
                """
            MPP X0*X1
            MPP Z0*Z1
        """
            ),
            q2i={0: 0, 1: 1},
            flows=[
                gen.Flow(end=zz, obs_key="a", mids=[1]),
                gen.Flow(end=xx, obs_key="b", mids=[0]),
            ],
        )
    )

    c.append(
        gen.Chunk(
            stim.Circuit(
                """
            MPP X0*X1
        """
            ),
            q2i={0: 0, 1: 1},
            discarded_inputs=[zz.with_name("a")],
            flows=[gen.Flow(start=xx, obs_key="b", mids=[0])],
        )
    )

    assert c.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        MPP X0*X1 Z0*Z1
        OBSERVABLE_INCLUDE(0) rec[-2]
        TICK
        MPP X0*X1
        OBSERVABLE_INCLUDE(0) rec[-1]
    """
    )


def test_chunk_negative_index_flow_measurement():
    chunk = gen.Chunk(
        circuit=stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            QUBIT_COORDS(1, 0) 1
            QUBIT_COORDS(2, 0) 2
            R 0 1 2
            M 0 1 2
        """
        ),
        flows=[
            gen.Flow(mids=[-1], center=0),
            gen.Flow(mids=[-2], center=0),
            gen.Flow(mids=[-3], center=0),
        ],
    )
    compiler = gen.ChunkCompiler()
    compiler.append(chunk)
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        R 0 1 2
        M 0 1 2
        DETECTOR(0, 0, 0) rec[-1]
        DETECTOR(0, 0, 1) rec[-2]
        DETECTOR(0, 0, 2) rec[-3]
    """
    )


def test_merge_ticks():
    q2i = {k: k for k in range(3)}
    init_chunk = gen.Chunk(
        q2i=q2i,
        circuit=stim.Circuit(
            """
            R 0 2
            TICK
            """
        ),
        flows=[gen.Flow(end=gen.PauliMap.from_zs([0, 2]))],
        wants_to_merge_with_next=True,
    )
    rep_chunk = gen.Chunk(
        q2i=q2i,
        circuit=stim.Circuit(
            """
            R 1
            TICK
            CX 0 1
            TICK
            CX 2 1
            TICK
            M 1
            """
        ),
        flows=[
            gen.Flow(start=gen.PauliMap.from_zs([0, 2]), mids=[0]),
            gen.Flow(end=gen.PauliMap.from_zs([0, 2]), mids=[-1]),
        ],
    )
    measure_chunk = init_chunk.time_reversed()

    compiler = gen.ChunkCompiler()
    compiler.append(init_chunk)
    compiler.append(rep_chunk)
    compiler.append(rep_chunk)
    compiler.append(measure_chunk)
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        R 0 2 1
        TICK
        CX 0 1
        TICK
        CX 2 1
        TICK
        M 1
        DETECTOR(1, 0, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        R 1
        TICK
        CX 0 1
        TICK
        CX 2 1
        TICK
        M 1
        DETECTOR(1, 0, 0) rec[-2] rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        M 2 0
        DETECTOR(1, 0, 0) rec[-3] rec[-2] rec[-1]
    """
    )


def test_preserves_tags():
    compiler = gen.ChunkCompiler()
    builder = gen.ChunkBuilder()
    builder.append("R", [0])
    builder.append("TICK")
    builder.append("X", [0], tag="test")
    builder.append("TICK")
    flow = gen.Flow(end=gen.PauliMap({0: "Z"}), center=2)
    compiler.append(gen.Chunk(builder.circuit, flows=[flow]))
    compiler.append_magic_end_chunk()
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        R 0
        TICK
        X[test] 0
        TICK
        MPP Z0
        DETECTOR(1, 0, 0) rec[-1]
    """
    )


def test_merges_with_loop():
    compiler = gen.ChunkCompiler()
    s = gen.Chunk(
        circuit=stim.Circuit(
            """
            QUBIT_COORDS(0, 0) 0
            R 0
        """
        ),
        flows=[gen.Flow(end=gen.PauliMap.from_zs([0]))],
        wants_to_merge_with_next=True,
    )
    compiler.append(s)
    compiler.append(
        gen.Chunk(
            circuit=stim.Circuit(
                """
            QUBIT_COORDS(0, 1) 0
            R 0
            M 0
        """
            ),
            flows=[
                gen.Flow(start=gen.PauliMap.from_zs([0]), end=gen.PauliMap.from_zs([0])),
                gen.Flow(mids=[0]),
            ],
        )
        * 5
    )
    compiler.append(s.time_reversed())
    assert compiler.finish_circuit() == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        R 0 1
        M 1
        DETECTOR(0, 0, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        TICK
        REPEAT 3 {
            R 1
            M 1
            DETECTOR(0, 0, 0) rec[-1]
            SHIFT_COORDS(0, 0, 1)
            TICK
        }
        R 1
        M 1
        DETECTOR(0, 0, 0) rec[-1]
        SHIFT_COORDS(0, 0, 1)
        M 0
        DETECTOR(0, 0, 0) rec[-1]
    """
    )
