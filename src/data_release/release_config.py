"""
Instrument registry for the data release.

One entry per released product. Each entry names the instrument, its model and
serial number, deployment location(s), sampling cadence, the burns for which
data exist, and the output filename. The registry is the single source of truth
consumed by the exporters, the manifest, and the README generator.

Author: Nathan Lima
Created: 2026-08-06
"""

# Location labels used verbatim in the released files and README.
BEDROOM = "Bedroom 2"
MORNING_ROOM = "Morning Room"

# The two MODULAIR-PM units were installed in the test house only after burn3.
# Portal records exist for earlier April dates from a different location; those
# rows are dropped from the release. First test-house burn is burn4 (2024-05-09),
# so the cutoff is the start of that calendar day.
MODULAIR_DEPLOY_START = "2024-05-09"

# Per-instrument analysis time shifts (minutes) applied elsewhere in the repo
# for synchronization. Released files keep the NATIVE clock; these values are
# documented in the README so a user can reproduce the synchronized analysis.
TIME_SHIFTS_MIN = {
    "AeroTrak Bedroom 2": 2.16,
    "AeroTrak Morning Room": 5.0,
    "DustTrak": 7.0,
    "MODULAIR-PM Bedroom 2": -2.97,
    "MODULAIR-PM Morning Room": 0.0,
    "SMPS": 0.0,
    "PurpleAir": 0.0,
}

# Ordered registry. 'kind' selects the exporter; 'config_key' resolves the
# source folder through data_config.json.
RELEASE_PRODUCTS = [
    {
        "id": "smps_bedroom2_number",
        "kind": "smps",
        "instrument": "SMPS",
        "model": "TSI SMPS 3938",
        "serial_number": "not recorded",
        "config_key": "smps",
        "location": BEDROOM,
        "cadence": "~135 s per scan",
        "burns": "1-10",
        "output": "SMPS_bedroom2_number.csv",
    },
    {
        "id": "dusttrak",
        "kind": "dusttrak",
        "instrument": "DustTrak DRX",
        "model": "TSI DustTrak DRX 8533",
        "serial_number": "8533163307",
        "config_key": "dusttrak",
        "location": f"{BEDROOM} (burns 1-6), {MORNING_ROOM} (burns 7-10)",
        "cadence": "1 min",
        "burns": "1-10",
        "output": "DustTrak_total_pm.csv",
    },
    {
        "id": "aerotrak_bedroom2",
        "kind": "aerotrak",
        "instrument": "AeroTrak 9306-V2",
        "model": "TSI AeroTrak 9306-V2 OPC",
        "serial_number": "not recorded",
        "config_key": "aerotrak_bedroom",
        "aerotrak_label": "AeroTrak1",
        "location": BEDROOM,
        "cadence": "1 min",
        "burns": "3-10",
        "output": "AeroTrak_bedroom2_counts.csv",
    },
    {
        "id": "aerotrak_morning_room",
        "kind": "aerotrak",
        "instrument": "AeroTrak 9306-V2",
        "model": "TSI AeroTrak 9306-V2 OPC",
        "serial_number": "not recorded",
        "config_key": "aerotrak_kitchen",
        "aerotrak_label": "AeroTrak2",
        "location": MORNING_ROOM,
        "cadence": "1 min",
        "burns": "2-10",
        "output": "AeroTrak_morning_room_counts.csv",
    },
    {
        "id": "modulair_bedroom2_portal",
        "kind": "modulair_portal",
        "instrument": "MODULAIR-PM (portal)",
        "model": "QuantAQ MODULAIR-PM (OPC-N3 + PMS5003)",
        "serial_number": "MOD-PM-00194",
        "config_key": "quantaq_bedroom",
        "location": BEDROOM,
        "cadence": "1 min (portal QA/QC product)",
        "burns": "4-10",
        "output": "MODULAIR-PM_bedroom2_portal_1min.csv",
    },
    {
        "id": "modulair_morning_room_portal",
        "kind": "modulair_portal",
        "instrument": "MODULAIR-PM (portal)",
        "model": "QuantAQ MODULAIR-PM (OPC-N3 + PMS5003)",
        "serial_number": "MOD-PM-00197",
        "config_key": "quantaq_kitchen",
        "location": MORNING_ROOM,
        "cadence": "1 min (portal QA/QC product)",
        "burns": "4-10",
        "output": "MODULAIR-PM_morning_room_portal_1min.csv",
    },
    {
        "id": "modulair_bedroom2_5s",
        "kind": "modulair_5s",
        "instrument": "MODULAIR-PM (raw 5 s)",
        "model": "QuantAQ MODULAIR-PM (OPC-N3 + PMS5003)",
        "serial_number": "MOD-PM-00194",
        "unit": "MODULAIR-PM1",
        "location": BEDROOM,
        "cadence": "5 s (raw SD-card record, no QA/QC)",
        "burns": "4-10",
        "output": "MODULAIR-PM_bedroom2_raw_5s.csv",
    },
    {
        "id": "modulair_morning_room_5s",
        "kind": "modulair_5s",
        "instrument": "MODULAIR-PM (raw 5 s)",
        "model": "QuantAQ MODULAIR-PM (OPC-N3 + PMS5003)",
        "serial_number": "MOD-PM-00197",
        "unit": "MODULAIR-PM2",
        "location": MORNING_ROOM,
        "cadence": "5 s (raw SD-card record, no QA/QC)",
        "burns": "4-10",
        "output": "MODULAIR-PM_morning_room_raw_5s.csv",
    },
    {
        "id": "purpleair_morning_room",
        "kind": "purpleair",
        "instrument": "PurpleAir PA-II",
        "model": "PurpleAir PA-II-SD",
        "serial_number": "not recorded",
        "config_key": "purpleair",
        "location": MORNING_ROOM,
        "cadence": "2 min",
        "burns": "6-10",
        "output": "PurpleAir_morning_room_pm25.csv",
    },
]


def product_by_id(product_id: str) -> dict:
    """Return the registry entry with the given id, or raise KeyError."""
    for p in RELEASE_PRODUCTS:
        if p["id"] == product_id:
            return p
    raise KeyError(f"No release product with id '{product_id}'.")
