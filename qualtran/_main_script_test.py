import subprocess
import sys

from qualtran import QInt, QUInt
from qualtran.__main__ import main


def test_main_smoke(capsys):
    """A smoketest for the CLI to make sure it runs without error."""
    main(argv=["-c", "-o", "outfile.o", "qualtran.bloqs.arithmetic.Add(a_dtype=QUInt(5))"])
    captured = capsys.readouterr()
    assert "Compile flag: True" in captured.out
    assert "Output file: outfile.o" in captured.out
    assert "Parsed Bloq: BloqCode" in captured.out


def test_main_subprocess():
    """A smoketest for the CLI using a subprocess."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "qualtran",
            "-c",
            "-o",
            "outfile.o",
            "qualtran.bloqs.arithmetic.multiplication.Square(bitsize=5)",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Compile flag: True" in result.stdout
    assert "Output file: outfile.o" in result.stdout
    assert "Parsed Bloq: BloqCode" in result.stdout
