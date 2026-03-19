"""
Data path resolution for WUI fire research multi-instrument, multi-machine setup.
Handles instrument-specific data folders with multiple files per instrument.

This module provides a portable way to access data files across different machines
(desktop, laptop) without hardcoding absolute paths in analysis scripts.

Author: Nathan Lima
Date: 2025-12-17
Institution: National Institute of Standards and Technology (NIST)
"""

import json
import socket
import sys
from pathlib import Path
from typing import List


class WUIDataPathResolver:
    """Resolves data paths for WUI research instruments across machines."""

    def __init__(self, config_file="data_config.json"):
        self.repo_root = Path(__file__).parent.parent.resolve()
        self.config_file = self.repo_root / config_file
        self.machine_name = socket.gethostname()
        self.config = self._load_config()

    def _load_config(self):
        """Load configuration file with helpful error messages."""
        if not self.config_file.exists():
            error_msg = f"""
╔════════════════════════════════════════════════════════════════╗
║  DATA CONFIGURATION NOT FOUND                                  ║
╚════════════════════════════════════════════════════════════════╝

Configuration file missing: {self.config_file}

SETUP INSTRUCTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Copy the template to create your local config
    copy data_config.template.json data_config.json

Step 2: Edit data_config.json with YOUR machine's data paths
    notepad data_config.json

Step 3: Example configuration for this machine ({self.machine_name}):

{{
  "machine_name": "{self.machine_name}",
  "data_root": "C:/Users/<user_name>/Documents/NIST/WUI_smoke",

  "instruments": {{
    "aerotrak_bedroom": {{
      "path": "C:/Users/<user_name>/Documents/NIST/WUI_smoke/burn_data/aerotraks/bedroom2",
      "description": "AeroTrak particle counter - Bedroom2",
      "file_pattern": "*.xlsx"
    }},
    "aerotrak_kitchen": {{
      "path": "C:/Users/<user_name>/Documents/NIST/WUI_smoke/burn_data/aerotraks/kitchen",
      "description": "AeroTrak particle counter - Kitchen",
      "file_pattern": "*.xlsx"
    }},
    "dusttrak": {{
      "path": "C:/Users/<user_name>/Documents/NIST/WUI_smoke/burn_data/dusttrak",
      "description": "DustTrak aerosol monitor",
      "file_pattern": "*.xlsx"
    }}
  }},

  "common_folders": {{
    "burn_log": "C:/Users/<user_name>/Documents/NIST/WUI_smoke/burn_log.xlsx",
    "output_figures": "C:/Users/<user_name>/Documents/NIST/WUI_smoke/Paper_figures"
  }}
}}

Step 4: Run setup verification
    python -c "from src.data_paths import resolver; resolver.list_instruments()"

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
            print(error_msg, file=sys.stderr)
            raise FileNotFoundError(error_msg)

        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"[OK] Loaded WUI data config for: {config.get('machine_name', 'unknown')}")
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in {self.config_file}: {e}") from e

    def get_instrument_path(self, instrument: str) -> Path:
        """
        Get the data folder path for a specific instrument.

        Parameters
        ----------
        instrument : str
            Instrument name (e.g., 'aerotrak_bedroom', 'smps', 'dusttrak')

        Returns
        -------
        pathlib.Path
            Resolved absolute path to instrument data folder

        Examples
        --------
        >>> resolver = WUIDataPathResolver()
        >>> aerotrak_path = resolver.get_instrument_path('aerotrak_bedroom')
        >>> data_file = aerotrak_path / 'all_data.xlsx'
        """
        instruments = self.config.get("instruments", {})

        if instrument not in instruments:
            available = list(instruments.keys())
            raise KeyError(
                f"Instrument '{instrument}' not configured.\n"
                f"Available instruments: {available}\n"
                f"Update your data_config.json to add this instrument."
            )

        path_str = instruments[instrument]["path"]
        path = Path(path_str).resolve()

        if not path.exists():
            print(f"[WARNING] Instrument path does not exist: {path}", file=sys.stderr)
            print("          Create folder or update path in data_config.json", file=sys.stderr)

        return path

    def get_instrument_files(self, instrument: str, pattern: str | None = None) -> List[Path]:
        """
        Get all data files for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument name
        pattern : str, optional
            File pattern (e.g., '*.csv'). If None, uses config default.

        Returns
        -------
        list of Path
            List of data files found

        Examples
        --------
        >>> resolver = WUIDataPathResolver()
        >>> smps_files = resolver.get_instrument_files('smps', '*_MassConc.xlsx')
        >>> for f in smps_files:
        ...     print(f.name)
        """
        inst_path = self.get_instrument_path(instrument)

        # Get pattern from config if not provided
        search_pattern = (
            pattern
            if pattern is not None
            else self.config["instruments"][instrument].get("file_pattern", "*.*")
        )

        files = sorted(inst_path.glob(search_pattern))

        if not files:
            print(
                f"[WARNING] No files found matching '{search_pattern}' in {inst_path}",
                file=sys.stderr,
            )

        return files

    def get_common_file(self, file_key: str) -> Path:
        """
        Get path to common files (burn_log, decay dates, etc.).

        Parameters
        ----------
        file_key : str
            Common file key (e.g., 'burn_log', 'output_figures')

        Returns
        -------
        pathlib.Path
            Resolved path

        Examples
        --------
        >>> resolver = WUIDataPathResolver()
        >>> burn_log = resolver.get_common_file('burn_log')
        >>> df = pd.read_excel(burn_log, sheet_name='Sheet2')
        """
        common = self.config.get("common_folders", {})

        if file_key not in common:
            available = list(common.keys())
            raise KeyError(f"Common file '{file_key}' not found. Available: {available}")

        return Path(common[file_key]).resolve()

    def list_instruments(self):
        """Print all configured instruments and their status."""
        print("\n" + "=" * 100)
        print("WUI Research Data Configuration")
        print("=" * 100)
        print(f"\nMachine: {self.machine_name}")
        print(f"Config:  {self.config_file}")

        instruments = self.config.get("instruments", {})
        if instruments:
            print(f"\n{'Instrument':<20} | Status | Files | Path")
            print("-" * 100)

            for name, info in instruments.items():
                path = Path(info["path"])
                exists = "OK" if path.exists() else "MISS"

                if path.exists():
                    pattern = info.get("file_pattern", "*.*")
                    file_count = len(list(path.glob(pattern)))
                    file_info = f"{file_count:>5}"
                else:
                    file_info = "  N/A"

                # Truncate long paths for display
                path_str = str(path)
                if len(path_str) > 60:
                    path_str = "..." + path_str[-57:]

                print(f"{name:<20} | {exists:^6} | {file_info} | {path_str}")
        else:
            print("\n[WARNING] No instruments configured!")

        common = self.config.get("common_folders", {})
        if common:
            print(f"\n{'Common File':<25} | Status | Path")
            print("-" * 100)
            for name, path_str in common.items():
                path = Path(path_str)
                exists = "OK" if path.exists() else "MISS"

                # Truncate long paths for display
                if len(path_str) > 65:
                    path_str = "..." + path_str[-62:]

                print(f"{name:<25} | {exists:^6} | {path_str}")

        print()

    def get_data_root(self) -> Path:
        """Get the root data directory."""
        return Path(self.config.get("data_root", self.repo_root / "data")).resolve()


# Convenience instance for easy importing
resolver = WUIDataPathResolver()


# Convenience functions
def get_instrument_path(instrument: str) -> Path:
    """Get data folder path for an instrument."""
    return resolver.get_instrument_path(instrument)


def get_instrument_files(instrument: str, pattern: str | None = None) -> List[Path]:
    """Get all data files for an instrument."""
    return resolver.get_instrument_files(instrument, pattern)


def get_common_file(file_key: str) -> Path:
    """Get path to common file."""
    return resolver.get_common_file(file_key)


def get_data_root() -> Path:
    """Get the data root directory."""
    return resolver.get_data_root()
