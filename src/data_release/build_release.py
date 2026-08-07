"""
Build the WUI PM data release.

Runs the per-instrument exporters, writes one tidy CSV per product to the MIDAS
release directory, and records a manifest (file, rows, date range, source
folder). Run under the project conda environment:

    CONDA_NO_PLUGINS=true conda run -n wui python -m src.data_release.build_release

Select a subset with --only (comma-separated product ids), e.g.

    ... build_release --only smps_bedroom2_number,dusttrak

No raw data are written or committed. Existing outputs are never overwritten
silently; a name collision is versioned with an ISO date suffix.

Author: Nathan Lima
Created: 2026-08-06
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data_release import exporters, release_config as rc  # noqa: E402
from src.data_release.readme import write_readme  # noqa: E402
from src.data_release.writers import release_dir, write_release_csv  # noqa: E402

# Map registry 'kind' to the exporter function. Only implemented kinds appear;
# unimplemented kinds are skipped with a notice so partial runs are explicit.
EXPORTERS = {
    "smps": exporters.prepare_smps,
    "dusttrak": exporters.prepare_dusttrak,
    "aerotrak": exporters.prepare_aerotrak,
    "modulair_portal": exporters.prepare_modulair_portal,
    "modulair_5s": exporters.prepare_modulair_5s,
    "purpleair": exporters.prepare_purpleair,
}


def build(only=None):
    """
    Run the release build.

    Parameters
    ----------
    only : list of str or None
        If given, only products whose id is in the list are built.

    Returns
    -------
    pd.DataFrame
        Manifest of written files.
    """
    today = date.today().isoformat()
    out_dir = release_dir()
    print(f"Release directory: {out_dir}")

    rows = []
    for product in rc.RELEASE_PRODUCTS:
        pid = product["id"]
        if only and pid not in only:
            continue

        exporter = EXPORTERS.get(product["kind"])
        if exporter is None:
            print(f"  [skip] {pid}: exporter for kind '{product['kind']}' "
                  f"not implemented yet.")
            continue

        print(f"\nBuilding {pid} ({product['instrument']}, {product['location']})")
        try:
            df = exporter(product)
        except Exception as exc:  # noqa: BLE001
            print(f"  [error] {pid}: {exc}")
            continue

        path = write_release_csv(df, product["output"], today)
        n_burn_rows = int((df["burn_id"] != "").sum())
        dmin = df["datetime"].min()
        dmax = df["datetime"].max()
        print(f"  wrote {path.name}: {len(df)} rows "
              f"({n_burn_rows} on burn days), {dmin} to {dmax}")

        rows.append({
            "product_id": pid,
            "file": path.name,
            "instrument": product["instrument"],
            "serial_number": product["serial_number"],
            "location": product["location"],
            "cadence": product["cadence"],
            "burns_available": product["burns"],
            "n_rows": len(df),
            "n_rows_on_burn_days": n_burn_rows,
            "datetime_min": dmin,
            "datetime_max": dmax,
        })

    manifest = pd.DataFrame(rows)
    if len(manifest):
        manifest_path = out_dir / "manifest.csv"
        # Manifest is regenerated each run; version if a prior one exists.
        if manifest_path.exists():
            manifest_path = out_dir / f"manifest_{today}.csv"
        manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
        print(f"\nManifest: {manifest_path.name} ({len(manifest)} products)")

        # Write the dataset README only for a full build; a --only subset would
        # produce a partial instrument table.
        if not only:
            readme_path = write_readme(out_dir, manifest, today)
            print(f"README: {readme_path.name}")
        else:
            print("README: skipped (subset build; run without --only to write it)")
    else:
        print("\nNo products built.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Build the WUI PM data release.")
    parser.add_argument(
        "--only",
        default=None,
        help="Comma-separated product ids to build (default: all implemented).",
    )
    args = parser.parse_args()
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    build(only=only)


if __name__ == "__main__":
    main()
