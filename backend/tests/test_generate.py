import csv
from collections import Counter

from facescore.generate import AGE_BANDS, ETHNICITIES, GENDERS, build_grid, run


def test_grid_balanced():
    n = len(GENDERS) * len(AGE_BANDS) * len(ETHNICITIES) * 4
    rows = build_grid(n)
    c = Counter((r["gender"], r["age_band"], r["ethnicity"]) for r in rows)
    assert len(c) == len(GENDERS) * len(AGE_BANDS) * len(ETHNICITIES)
    assert set(c.values()) == {4}
    assert len({r["id"] for r in rows}) == n


def test_dry_run_writes_ingestable_csv(tmp_path):
    meta = run(30, tmp_path / "pool", "dry")
    with open(meta) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 30
    assert {"id", "gender", "age_band", "ethnicity"} <= set(rows[0])
