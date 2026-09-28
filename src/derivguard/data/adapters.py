"""Source-specific adapters; raw bytes remain immutable and source semantics stay explicit."""

from __future__ import annotations

import io
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path

import pandas as pd


def iter_public_spx_sample(
    archive: Path,
    *,
    underlyings: Sequence[str] = ("SPX", "SPXW"),
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Stream daily SPX/SPXW frames without materializing the 806 MB archive."""
    selected = {item.upper() for item in underlyings}
    with zipfile.ZipFile(archive) as bundle:
        members = sorted(
            name
            for name in bundle.namelist()
            if name.startswith("day_by_date/") and name.endswith("_options.csv")
        )
        for member in members:
            with bundle.open(member) as stream:
                frame = pd.read_csv(stream, low_memory=False)
            if "underlying" not in frame:
                raise ValueError(f"{member} has no underlying field")
            mask = frame["underlying"].astype("string").str.upper().isin(selected)
            yield member, frame.loc[mask].copy()


def read_cboe_marking_long(path: Path) -> pd.DataFrame:
    """Reshape Cboe's paired call/put marking rows into canonical long quotes."""
    wide = pd.read_csv(path, low_memory=False)
    required = {"root", "expiry", "strike", "underlying_symbol"}
    if missing := required - set(wide.columns):
        raise ValueError(f"Cboe marking file missing columns: {sorted(missing)}")
    records: list[pd.DataFrame] = []
    for kind in ("call", "put"):
        mapping = {
            "root": "underlying",
            "expiry": "expiration",
            "strike": "strike",
            "underlying_symbol": "underlying_symbol",
            f"{kind}_message_time": "quote_datetime",
            f"{kind}_final_indicative_bid": "bid",
            f"{kind}_final_indicative_ask": "ask",
            f"{kind}_final_indicative_bid_size": "bid_size",
            f"{kind}_final_indicative_ask_size": "ask_size",
            f"{kind}_underlying_value": "underlying_price",
            f"{kind}_osi_identifier": "contract",
        }
        if missing := set(mapping) - set(wide.columns):
            raise ValueError(f"Cboe marking {kind} fields missing: {sorted(missing)}")
        side = wide[list(mapping)].rename(columns=mapping)
        side["option_type"] = kind
        side["quote_date"] = pd.to_datetime(side["quote_datetime"], errors="coerce").dt.date
        side["observation_class"] = "CBOE_INDICATIVE_MARK_NOT_OPRA_NBBO"
        records.append(side)
    return pd.concat(records, ignore_index=True)


def iter_datashop_nested_csv(archive: Path) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield CSV frames from the official sample's nested ZIP structure."""
    with zipfile.ZipFile(archive) as outer:
        inner_names = sorted(name for name in outer.namelist() if name.lower().endswith(".zip"))
        for inner_name in inner_names:
            with outer.open(inner_name) as compressed:
                payload = compressed.read()
            with zipfile.ZipFile(io.BytesIO(payload)) as inner:
                csv_names = sorted(
                    name for name in inner.namelist() if name.lower().endswith(".csv")
                )
                for csv_name in csv_names:
                    with inner.open(csv_name) as stream:
                        yield f"{inner_name}!{csv_name}", pd.read_csv(stream, low_memory=False)
