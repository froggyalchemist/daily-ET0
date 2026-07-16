"""
cmip6_archive.py: Locate, open, and validate CMIP6 NetCDF files from a local archive.
"""

import glob
import re
from rich.console import Console
from rich import print              # pretty print in the terminal
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import xarray as xr


@dataclass(frozen=True) # frozen = can't modify once created
class GCMConfig:
    """Represents a single GCM ensemble member."""
    name: str
    variant_id: str
    grid_label: str


GCM_REGISTRY = [
    GCMConfig("IPSL-CM6A-LR",  "r1i1p1f1",  "gr"),
    GCMConfig("MIROC6",         "r1i1p1f1",  "gn"),
    GCMConfig("MRI-ESM2-0",     "r1i1p1f1",  "gn"),
    GCMConfig("NorESM2-LM",     "r1i1p1f1",  "gn"),
    GCMConfig("NorESM2-MM",     "r1i1p1f1",  "gn"),
    GCMConfig("MPI-ESM1-2-HR",  "r1i1p1f1",  "gn"),
    GCMConfig("MPI-ESM1-2-LR",  "r1i1p1f1", "gn"),
    GCMConfig("UKESM1-0-LL",    "r1i1p1f2",  "gn"),
]

EXPERIMENTS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]
VARIABLES   = ["hfls", "hfss", "hurs", "ps", "sfcWind", "tas"]

# Required start and end years all experiments except 'historical'
REQUIRED_YEAR_START = 2015
REQUIRED_YEAR_END   = 2100

# Matches the date range at the end of a CMIP6 filename, e.g. 20150101-20991231
_DATE_RANGE_RE = re.compile(r"(\d{4})\d{4}-(\d{4})\d{4}\.nc$")

class CMIP6LocalArchive:
    """Represents a local CMIP6 archive nested as <root>/<gcm>/<expid>/<varid>/."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"CMIP6LocalArchive(root='{self.root}')"

    def get_paths_for(self, gcm: str, expid: str, varid: str) -> list[Path]:
        """Return sorted list of NetCDF paths for a GCM / experiment / variable."""
        pattern = (
            self.root / "**" / gcm / expid / "**" / varid
            / "**" / f"{varid}_*_{gcm}_{expid}*.nc"
        )
        matches = sorted(glob.glob(str(pattern), recursive=True)) # sort chronologically
        if not matches:
            raise FileNotFoundError(f"No files for {gcm} / {expid} / {varid}")
        return [Path(m) for m in matches]

    def get_variable_dataset(
        self, gcm: str, expid: str, varid: str, chunks: dict | None = None
    ) -> xr.Dataset:
        """Open a dataset for a GCM / experiment / variable, handling multi-file cases."""
        paths = self.get_paths_for(gcm, expid, varid)
        if len(paths) == 1:
            return xr.open_dataset(paths[0], chunks=chunks, data_vars='all')
        return xr.open_mfdataset(paths, combine="by_coords", chunks=chunks, data_vars='all')


def get_year_coverage_from_paths(paths: list[Path]) -> tuple[int, int] | None:
    """
    Parse start/end years from CMIP6 filenames without opening any files.
    Returns (min_start_year, max_end_year), or None if no date range is found.
    """
    start_years, end_years = [], []
    for p in paths:
        m = _DATE_RANGE_RE.search(p.name)
        if m:
            start_years.append(int(m.group(1)))
            end_years.append(int(m.group(2)))
    if not start_years:
        return None
    return min(start_years), max(end_years)


def check_all_data(
    archive: CMIP6LocalArchive,
    gcm_registry: list[GCMConfig] = GCM_REGISTRY,
    experiments: list[str] = EXPERIMENTS,
    variables: list[str] = VARIABLES,
) -> pd.DataFrame:
    """
    Validate file availability and time coverage for all GCM × experiment × variable combinations.
    Returns a DataFrame with columns: gcm, experiment, variable, n_files, status, coverage, error.
    """
    rows = []

    # Loop through all GCM × experiment × variable combinations
    for gcm in gcm_registry:
        for exp in experiments:
            for var in variables:
                row = dict(gcm=gcm.name, experiment=exp, variable=var,
                           n_files=None, status=None, coverage=None, error=None)
                try:
                    # Try opening the variable's file
                    paths = archive.get_paths_for(gcm.name, exp, var)
                    row["n_files"] = len(paths)

                    # Read years from file name (e.g. xxxx_20152100.nc)
                    year_range = get_year_coverage_from_paths(paths)
                    if year_range is None:
                        row["status"] = "⚠️ Unparseable filenames"
                    else:
                        actual_start, actual_end = year_range
                        row["coverage"] = f"{actual_start}–{actual_end}"

                        # If experiment is historical, coverage must be from 1850 to 2014
                        if exp == "historical":
                            required_start = 1850
                            required_end = 2014
                        # For other experiments it is from REQUIRED_YEAR_START to REQUIRED_YEAR_END
                        else:
                            required_start = REQUIRED_YEAR_START
                            required_end = REQUIRED_YEAR_END

                        # Check if file covers required date range
                        if actual_start > required_start:
                            row["status"] = "⚠️ Starts late"
                        elif actual_end < required_end:
                            row["status"] = "⚠️ Ends early"
                        else:
                            row["status"] = "✅ Complete"

                except FileNotFoundError as e:
                    row["status"] = "☠️ Missing"
                    row["error"]  = str(e)
                except Exception as e:
                    row["status"] = "❓ Unknown error"
                    row["error"]  = f"{type(e).__name__}: {e}"
                rows.append(row)

    return pd.DataFrame(rows)


if __name__ == "__main__":

    console = Console()

    # 'archive' lets us open datasets without having to type the full path
    archive = CMIP6LocalArchive(root="/work10/archive/CMIP6/CMIP-SSPs/")
    print("Created local archive: ", archive, "\n")

    # Small test to see if we can open a dataset
    #test = GCMConfig("MPI-ESM1-2-HR", "r1i1p1f1", "gn")
    #ds = archive.get_variable_dataset(test.name, "ssp126", "ps")
    #print(ds["ps"], "\n")
    #ds.close()

    # Full validation sweep
    with console.status("[bold magenta]Checking local archive contains all GCM × experiment × variable combinations:\n", spinner='clock') as status:
        df = check_all_data(archive)
        with pd.option_context("display.max_rows", None, "display.max_columns", None):
            print(df.drop(columns=['error']))

    # Print summary and problems
    print("\n[bold magenta]Summary:\n")
    print(pd.DataFrame(df["status"].value_counts()))

    problems = df[df["status"] != "✅ Complete"]
    if problems.empty:
        print("\nAll combinations complete. 🎉")
    else:
        print(f"\n{len(problems)} problem(s):\n")
        print(problems[["gcm", "experiment", "variable", "status", "coverage", "error"]])
