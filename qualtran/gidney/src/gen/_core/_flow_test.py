import gen


def test_with_xz_flipped():
    assert gen.Flow(start=gen.PauliMap({1: "X", 2: "Z"}), center=0).with_xz_flipped() == gen.Flow(
        start=gen.PauliMap({1: "Z", 2: "X"}), center=0
    )
