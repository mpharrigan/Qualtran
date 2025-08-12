import stim

import gen


def test_builder_init():
    builder = gen.ChunkBuilder([0, 1j, 3 + 2j])
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(0, 1) 1
        QUBIT_COORDS(3, 2) 2
        """
    )


def test_append_tick():
    builder = gen.ChunkBuilder([0])
    builder.append("TICK")
    builder.append("TICK")
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        TICK
        TICK
        """
    )


def test_append_shift_coords():
    builder = gen.ChunkBuilder([0])
    builder.append("SHIFT_COORDS", arg=[0, 0, 1])
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        SHIFT_COORDS(0, 0, 1)
        """
    )


def test_append_measurements():
    builder = gen.ChunkBuilder(range(6))

    builder.append("MXX", [(2, 3)])
    assert builder.lookup_mids([(2, 3)]) == [0]
    assert builder.lookup_mids([(3, 2)]) == [0]

    builder.append("MYY", [(5, 4)])
    assert builder.lookup_mids([(4, 5)]) == [1]
    assert builder.lookup_mids([(5, 4)]) == [1]

    builder.append("M", [3])
    assert builder.lookup_mids([3]) == [2]


def test_append_measurements_canonical_order():
    builder = gen.ChunkBuilder(range(6))

    builder.append("MX", [5, 2, 3])
    assert builder.lookup_mids([2]) == [0]
    assert builder.lookup_mids([3]) == [1]
    assert builder.lookup_mids([5]) == [2]

    builder.append("MZZ", [(5, 2), (3, 4)])
    assert builder.lookup_mids([(2, 5)]) == [3]
    assert builder.lookup_mids([(3, 4)]) == [4]

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        QUBIT_COORDS(4, 0) 4
        QUBIT_COORDS(5, 0) 5
        MX 2 3 5
        MZZ 2 5 3 4
        """
    )


def test_append_mpp():
    builder = gen.ChunkBuilder([2 + 3j, 5 + 7j, 11 + 13j])

    xxx = gen.PauliMap.from_xs([2 + 3j, 5 + 7j, 11 + 13j])
    z_z = gen.PauliMap.from_zs([11 + 13j, 2 + 3j])
    builder.append("MPP", [xxx, z_z])
    assert builder.lookup_mids([xxx]) == [0]
    assert builder.lookup_mids([z_z]) == [1]

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(5, 7) 1
        QUBIT_COORDS(11, 13) 2
        MPP X0*X1*X2 Z0*Z2
        """
    )


def test_append_observable_include():
    builder = gen.ChunkBuilder([2 + 3j, 5 + 7j, 11 + 13j])

    builder.append("R", [5 + 7j])
    builder.append("M", [2 + 3j, 5 + 7j, 11 + 13j], measure_key_func=lambda e: (e, "X"))
    builder.append("OBSERVABLE_INCLUDE", [(5 + 7j, "X")], arg=2)

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(5, 7) 1
        QUBIT_COORDS(11, 13) 2
        R 1
        M 0 1 2
        OBSERVABLE_INCLUDE(2) rec[-2]
    """
    )


def test_append_detector():
    builder = gen.ChunkBuilder([2 + 3j, 5 + 7j, 11 + 13j])

    builder.append("R", [5 + 7j])
    builder.append("M", [2 + 3j, 5 + 7j, 11 + 13j], measure_key_func=lambda e: (e, "X"))
    builder.append("DETECTOR", [(5 + 7j, "X")], arg=[2, 3, 5])

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(2, 3) 0
        QUBIT_COORDS(5, 7) 1
        QUBIT_COORDS(11, 13) 2
        R 1
        M 0 1 2
        DETECTOR(2, 3, 5) rec[-2]
        """
    )


def test_make_surface_code_first_round():
    diameter = 3
    tiles = []

    for x in range(-1, diameter):
        for y in range(-1, diameter):
            m = x + 1j * y + 0.5 + 0.5j
            potential_data = [m + 1j**k * (0.5 + 0.5j) for k in range(4)]
            data = [d for d in potential_data if 0 <= d.real < diameter if 0 <= d.imag < diameter]
            if len(data) not in [2, 4]:
                continue

            basis = "XZ"[(x.real + y.real) % 2 == 0]
            if not (0 <= m.real < diameter - 1) and basis != "Z":
                continue
            if not (0 <= m.imag < diameter - 1) and basis != "X":
                continue
            tiles.append(gen.Tile(measure_qubit=m, data_qubits=data, bases=basis))

    patch = gen.Patch(tiles)
    obs_x = gen.PauliMap({q: "X" for q in patch.data_set if q.real == 0}).with_name("LX")
    obs_z = gen.PauliMap({q: "Z" for q in patch.data_set if q.imag == 0}).with_name("LZ")
    code = gen.StabilizerCode(patch, logicals=[(obs_x, obs_z)]).with_transformed_coords(
        lambda e: e * (1 - 1j)
    )

    builder = gen.ChunkBuilder(code.used_set)

    mxs = {tile.measure_qubit for tile in code.patch if tile.basis == "X"}
    mzs = {tile.measure_qubit for tile in code.patch if tile.basis == "Z"}
    builder.append("RX", mxs)
    builder.append("RZ", mzs | code.data_set)
    builder.append("TICK")

    for layer in range(4):
        offset = [1j, 1, -1, -1j][layer]
        cxs = []
        for tile in code.tiles:
            m = tile.measure_qubit
            s = -1 if tile.basis == "Z" else +1
            d = m + offset * (s if 1 <= layer <= 2 else 1)
            if d in code.data_set:
                cxs.append((m, d)[::s])
        builder.append("CX", cxs)
        builder.append("TICK")
    builder.append("MX", mxs)
    builder.append("MZ", mzs)
    for z in gen.sorted_complex(mzs):
        builder.append("DETECTOR", [z], arg=[z.real, z.imag, 0])

    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, -1) 0
        QUBIT_COORDS(0, 0) 1
        QUBIT_COORDS(1, -1) 2
        QUBIT_COORDS(1, 0) 3
        QUBIT_COORDS(1, 1) 4
        QUBIT_COORDS(1, 2) 5
        QUBIT_COORDS(2, -2) 6
        QUBIT_COORDS(2, -1) 7
        QUBIT_COORDS(2, 0) 8
        QUBIT_COORDS(2, 1) 9
        QUBIT_COORDS(2, 2) 10
        QUBIT_COORDS(3, -2) 11
        QUBIT_COORDS(3, -1) 12
        QUBIT_COORDS(3, 0) 13
        QUBIT_COORDS(3, 1) 14
        QUBIT_COORDS(4, 0) 15
        QUBIT_COORDS(4, 1) 16
        RX 0 7 9 16
        R 1 2 3 4 5 6 8 10 11 12 13 14 15
        TICK
        CX 0 1 4 3 7 8 9 10 12 11 14 13
        TICK
        CX 0 2 1 3 6 11 7 12 8 13 9 14
        TICK
        CX 7 2 8 3 9 4 10 5 15 13 16 14
        TICK
        CX 2 3 4 5 7 6 9 8 12 13 16 15
        TICK
        MX 0 7 9 16
        M 3 5 11 13
        DETECTOR(1, 0, 0) rec[-4]
        DETECTOR(1, 2, 0) rec[-3]
        DETECTOR(3, -2, 0) rec[-2]
        DETECTOR(3, 0, 0) rec[-1]
        """
    )


def test_skip_unknown_1qm():
    builder = gen.ChunkBuilder(allowed_qubits=[0, 1, 2, 3], unknown_qubit_append_mode="skip")
    builder.append("M", [2, -1, 1, 25, 3])
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        M 1 2 3
    """
    )
    assert builder.lookup_mids([1]) == [0]
    assert builder.lookup_mids([2]) == [1]
    assert builder.lookup_mids([3]) == [2]


def test_skip_unknown_2qm():
    builder = gen.ChunkBuilder(allowed_qubits=[0, 1, 2, 3], unknown_qubit_append_mode="skip")
    builder.append("MZZ", [(2, 3), (-1, 5), (0, 1)])
    assert builder.circuit == stim.Circuit(
        """
        QUBIT_COORDS(0, 0) 0
        QUBIT_COORDS(1, 0) 1
        QUBIT_COORDS(2, 0) 2
        QUBIT_COORDS(3, 0) 3
        MZZ 0 1 2 3
    """
    )
    assert builder.lookup_mids([(0, 1)]) == builder.lookup_mids([(1, 0)]) == [0]
    assert builder.lookup_mids([(2, 3)]) == builder.lookup_mids([(3, 2)]) == [1]
