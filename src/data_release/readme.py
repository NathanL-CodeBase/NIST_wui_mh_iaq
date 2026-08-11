"""
README generator for the WUI PM data release.

Builds README_dataset.md from the instrument registry and the manifest produced
by the build. The text is written in the project's building-science voice: every
caveat is stated with the number that makes it checkable, and the known
instrument artifacts are documented rather than corrected in the data.

Author: Nathan Lima
Created: 2026-08-06
"""

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from src.data_release import release_config as rc  # noqa: E402


def _instrument_table(manifest: pd.DataFrame) -> str:
    """Return a Markdown table of the released products from the manifest."""
    header = (
        "| File | Instrument | Model | Serial | Location | Cadence | "
        "Burns | Rows | Date range |\n"
        "|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for _, r in manifest.iterrows():
        dmin = pd.to_datetime(r["datetime_min"]).strftime("%Y-%m-%d")
        dmax = pd.to_datetime(r["datetime_max"]).strftime("%Y-%m-%d")
        lines.append(
            f"| `{r['file']}` | {r['instrument']} | "
            f"{_model_for(r['product_id'])} | {r['serial_number']} | "
            f"{r['location']} | {r['cadence']} | {r['burns_available']} | "
            f"{r['n_rows']} | {dmin} to {dmax} |"
        )
    return header + "\n".join(lines) + "\n"


def _model_for(product_id: str) -> str:
    """Look up the instrument model string for a product id."""
    try:
        return rc.product_by_id(product_id)["model"]
    except KeyError:
        return ""


def build_readme(manifest: pd.DataFrame, generated_date: str) -> str:
    """
    Assemble the release README text.

    Parameters
    ----------
    manifest : pd.DataFrame
        Manifest returned by build_release.build().
    generated_date : str
        ISO date the release was generated.

    Returns
    -------
    str
        Markdown README content.
    """
    table = _instrument_table(manifest)

    return f"""# WUI manufactured-home smoke study: PM measurement data

Particulate matter (PM) measurements from ten controlled wood-crib smoke
infiltration burns in the NIST Indoor Air Quality Test House, a manufactured
home. Burns ran from 2024-04-26 (burn1) to 2024-05-31 (burn10). Each file holds
one instrument's record as measured, with no artifact corrections applied and
each instrument's native recorded clock preserved. Read the caveats below before
using any value.

Generated {generated_date}.

## Files

Every file shares five leading columns, then instrument-specific measurement
columns whose names carry their units.

- `datetime`: instrument's native local timestamp (see clock note below).
- `burn_id`: `burn1` through `burn10` when the row falls on a burn calendar day,
  empty otherwise. The full continuous deployment record is kept; the tag marks
  which rows belong to a burn day.
- `location`: `Bedroom 2` or `Morning Room`.
- `instrument`, `serial_number`: instrument label and serial (`not recorded`
  where the serial was not logged).

{table}

## Instruments and columns

**SMPS (TSI SMPS 3938), Bedroom 2, all ten burns.** Mobility size distributions
from 9.3 to 437 nm, one scan about every 135 s, reported as number concentration
(#/cm3). Columns are the diameter-midpoint bins (for example `14.13 nm (#/cm3)`),
the scan `Lower Size(nm)` and `Upper Size(nm)`, the distribution summary
statistics (D50, median, mean, geometric mean, mode, geometric standard
deviation), and `Total Concentration (#/cm3)`. This is the cleanest dataset and
the only one without a concentration-dependent optical artifact, but it is a
single point in Bedroom 2.

**AeroTrak 9306-V2 OPCs, Bedroom 2 and Morning Room, 1 min.** Optical particle
counts in six size channels, released as count concentration (#/cm3) named by
size range (`N 0.3-0.5um (#/cm3)` through `N 10-25um (#/cm3)`). Data are filtered
to rows the instrument flagged Flow Status = OK and Laser Status = OK. The
Bedroom 2 export had the entire Morning Room record merged into it for
2024-05-01 through 2024-05-08; those foreign rows were removed by matching
timestamp and all six differential counts to the Morning Room file, so the
Bedroom 2 file here starts 2024-05-04.

**DustTrak DRX 8533.** Photometric mass at 1 min. All five recorded channels are
released, converted from mg/m3 to ug/m3: `PM1 (ug/m3)`, `PM2.5 (ug/m3)`,
`PM4 (ug/m3)`, `PM10 (ug/m3)`, `TOTAL (ug/m3)`. See the DustTrak caveat below:
only `TOTAL` is usable. The instrument sat in Bedroom 2 for burns 1 to 6 and
moved to the Morning Room for burns 7 to 10.

**MODULAIR-PM portal product (QuantAQ MODULAIR-PM, OPC-N3 plus PMS5003),
Bedroom 2 (MOD-PM-00194) and Morning Room (MOD-PM-00197), 1 min, burns 4 to 10.**
The cloud-portal QA/QC product: mass (`PM1/PM2.5/PM10 (ug/m3)` and the OPC-N3
mass channels), OPC-N3 bin counts `bin0` through `bin23`, PMS5003 nephelometer
bins `neph_bin0` through `neph_bin5`, sample RH/temperature/pressure, and the
QA/QC `flag`. Rows before 2024-05-09 are excluded: the units logged earlier
April data at a different location before installation in the test house.

**MODULAIR-PM raw 5 s record, same two units, burns 4 to 10.** The
on-instrument SD-card record with no QA/QC filtering, logging one row every 5 s
(0.2 Hz, not 5 Hz). Columns are OPC-N3 `bin0` through `bin23`, nephelometer
`neph_bin0` through `neph_bin5`, and `flag`. The raw record recovers the peak
windows the portal product removes.

**PurpleAir PA-II, Morning Room, 2 min, burns 6 to 10.** PM2.5 from the two
internal channels, `PM2.5 A (ug/m3)` and `PM2.5 B (ug/m3)`, and their per-row
mean `PM2.5 Average (ug/m3)`.

## Timestamps

Released timestamps are each instrument's native recorded clock, not a
synchronized clock. The analysis behind the companion instrument paper applies
per-instrument shifts to align the records; to reproduce that alignment, add the
following to each instrument's timestamps:

| Instrument | Location | Shift to add (min) |
|---|---|---|
| AeroTrak 9306-V2 | Bedroom 2 | +2.16 |
| AeroTrak 9306-V2 | Morning Room | +5.0 |
| DustTrak DRX | (both) | +7.0 |
| MODULAIR-PM | Bedroom 2 | -2.97 |
| MODULAIR-PM | Morning Room | 0.0 |
| SMPS 3938 | Bedroom 2 | 0.0 |
| PurpleAir PA-II | Morning Room | 0.0 |

The MODULAIR-PM portal 1-minute value is a trailing 60 s average: the row stamped
at HH:MM:SS reports the mean of the preceding minute. To center it on the
5 s clock, subtract 30 s. The raw 5 s MODULAIR-PM timestamps are the
on-instrument UTC clock converted to local EDT by a fixed -4 h offset.

## Caveats

No gravimetric reference was run, so the dataset has no absolute mass anchor.
Co-located peak readings across instruments spread by up to 50-fold. The
companion instrument paper (in preparation) brackets the true peak mass rather
than trusting any single monitor.

The DustTrak reads 5 to 54 times the co-located instruments, an Arizona Road
Dust factory calibration applied to carbonaceous smoke. A size-fraction
configuration error left the PM1, PM2.5, PM4, and PM10 channels unusable; only
`TOTAL` carries meaning, and it still carries the photometric high bias.

Both optical particle counter types roll over at the peaks. Above roughly
900 ug/m3 the sub-micron counting channels collapse from coincidence, likely
compounded by the absorbing aerosol, so AeroTrak and OPC-N3 counts and derived
mass are unreliable through the densest part of each burn.

The MODULAIR-PM portal product applies an automated QA/QC filter that removes the
peak windows, producing a spurious cap near 560 ug/m3. The smallest OPC-N3 bin
also over-reads for a few hours after each peak. The raw 5 s records recover
both effects and carry no QA/QC removal.

## Disclaimer

Certain commercial equipment, instruments, software, or materials are identified in 
this repository in order to specify the experimental and analytical procedures 
adequately. Such identification is not intended to imply recommendation or 
endorsement of any product or service by NIST, nor is it intended to imply that the 
materials or equipment identified are necessarily the best available for the purpose.

## Provenance

Values are released as recorded. The only edits applied are unit conversions
(DustTrak mg/m3 to ug/m3; SMPS reported as #/cm3), the AeroTrak status filter and
Morning Room foreign-row removal described above, and the MODULAIR-PM pre-install
date exclusion. See `manifest.csv` for per-file row counts and date ranges.

Contact: Nathan M. Lima (nathan.lima@nist.gov); PI Dustin G. Poppendieck.
"""


def write_readme(out_dir: Path, manifest: pd.DataFrame, generated_date: str) -> Path:
    """
    Write README_dataset.md to the release directory (never silent overwrite).

    Parameters
    ----------
    out_dir : Path
        Release output directory.
    manifest : pd.DataFrame
        Manifest of written files.
    generated_date : str
        ISO date used for a name collision suffix.

    Returns
    -------
    Path
        Path written.
    """
    target = out_dir / "README_dataset.md"
    if target.exists():
        target = out_dir / f"README_dataset_{generated_date}.md"
    target.write_text(build_readme(manifest, generated_date), encoding="utf-8")
    return target
