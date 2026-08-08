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
VARIABLES   = ["rsds", "rsus", "rlds", "rlus", "hurs", "ps", "sfcWind", "tas"]

# Required start and end years all experiments except 'historical'
REQUIRED_YEAR_START = 2015
REQUIRED_YEAR_END   = 2100

# Required start and end years for the 'historical' experiment
HISTORICAL_YEAR_START = 1850
HISTORICAL_YEAR_END   = 2014

# Matches the date range at the end of a CMIP6 filename, e.g. 20150101-20991231
_DATE_RANGE_RE = re.compile(r"(\d{4})\d{4}-(\d{4})\d{4}\.nc$")


def get_gcm_config(name: str, registry: list[GCMConfig] = GCM_REGISTRY) -> GCMConfig:
    """Look up a GCMConfig by name in a registry, raising if it isn't found."""
    for config in registry:
        if config.name == name:
            return config
    raise ValueError(f"GCM '{name}' is not in the registry.")


def required_year_range(expid: str) -> tuple[int, int]:
    """Return the (start_year, end_year) of data actually needed for an experiment:
    REQUIRED_YEAR_START-REQUIRED_YEAR_END for ssps, HISTORICAL_YEAR_START-HISTORICAL_YEAR_END
    for historical."""
    if expid == "historical":
        return HISTORICAL_YEAR_START, HISTORICAL_YEAR_END
    return REQUIRED_YEAR_START, REQUIRED_YEAR_END


def _file_year_range(path: Path) -> tuple[int, int] | None:
    """Parse (start_year, end_year) from a single CMIP6 filename, or None if unparseable."""
    m = _DATE_RANGE_RE.search(path.name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


def _filter_paths_by_year_range(paths: list[Path], start: int, end: int) -> list[Path]:
    """Keep only the files whose date range overlaps [start, end], so we avoid opening
    (and dragging around in the task graph) years of data we don't actually need."""
    filtered = []
    for p in paths:
        year_range = _file_year_range(p)
        if year_range is None:
            # Can't tell from the filename -- keep it, better safe than sorry
            filtered.append(p)
            continue
        file_start, file_end = year_range
        if file_end >= start and file_start <= end:
            filtered.append(p)
    # Fall back to the full file list if filtering happened to remove everything
    return filtered if filtered else paths


class CMIP6LocalArchive:
    """Represents a local CMIP6 archive nested as <root>/<gcm>/<expid>/<varid>/."""

    def __init__(self, root: Path):
        self.root = Path(root)

    def __repr__(self) -> str:
        return f"CMIP6LocalArchive(root='{self.root}')"

    def get_paths_for(self, gcm: str, expid: str, varid: str,
                       registry: list[GCMConfig] = GCM_REGISTRY) -> list[Path]:
        """Return sorted list of NetCDF paths for a GCM / experiment / variable.

        Only files matching the variant_id registered for this GCM in `registry` are
        returned, so we never accidentally pick up a different ensemble member.
        """
        variant_id = get_gcm_config(gcm, registry).variant_id
        pattern = (
            self.root / "**" / gcm / expid / "**" / varid
            / "**" / f"{varid}_*_{gcm}_{expid}_{variant_id}*.nc"
        )
        matches = sorted(glob.glob(str(pattern), recursive=True)) # sort chronologically
        if not matches:
            raise FileNotFoundError(f"No files for {gcm} / {expid} / {varid} (variant {variant_id})")
        return [Path(m) for m in matches]

    def get_variable_dataset(self, gcm: str, expid: str, varid: str, chunks = "auto",
                              registry: list[GCMConfig] = GCM_REGISTRY) -> xr.Dataset:
        """Open a dataset for a GCM / experiment / variable, handling multi-file cases.

        Only opens the files needed to cover the required date range for `expid` and
        then trims the resulting dataset to exactly that range, so unused time steps
        are never opened or carried through the computation (see required_year_range).
        """
        paths = self.get_paths_for(gcm, expid, varid, registry=registry)

        start_year, end_year = required_year_range(expid)
        paths = _filter_paths_by_year_range(paths, start_year, end_year)

        if len(paths) == 1:
            ds = xr.open_dataset(paths[0], chunks=chunks)
        else:
            ds = xr.open_mfdataset(paths, combine="by_coords", chunks=chunks, data_vars='all', parallel=True)

        # Set the date range at the end of the file: trim to exactly what's required
        ds = ds.sel(time=slice(str(start_year), str(end_year)))

        return ds


def get_year_coverage_from_paths(paths: list[Path]) -> tuple[int, int] | None:
    """
    Parse start/end years from CMIP6 filenames without opening any files.
    Returns (min_start_year, max_end_year), or None if no date range is found.
    """
    start_years, end_years = [], []
    for p in paths:
        (start, end) = _file_year_range(p)
        if (start, end) is None:
            return None
        else:
            start_years.append(start)
            end_years.append(end)
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
                    paths = archive.get_paths_for(gcm.name, exp, var, registry=gcm_registry)
                    row["n_files"] = len(paths)

                    # Read years from file name (e.g. xxxx_20152100.nc)
                    year_range = get_year_coverage_from_paths(paths)
                    if year_range is None:
                        row["status"] = "⚠️ Unparseable filenames"
                    else:
                        actual_start, actual_end = year_range
                        row["coverage"] = f"{actual_start}–{actual_end}"

                        # If experiment is historical, coverage must be 
                        # from 1850 to 2014, else it's 2015 to 2100
                        required_start, required_end = required_year_range(exp)

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
    #test = archive.get_variable_dataset(gcm='MIROC6', expid='ssp126', varid="tas")
    #print(test)
    #test.close()

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
