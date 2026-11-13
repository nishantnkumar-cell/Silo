"""Broad sector for each UK candidate, for sector-relative signals.

Hand-kept from today's classifications, grouped coarsely (about a dozen
groups) so most groups hold several names. Applying today's labels to the
past is a mild look-ahead: a company that changed business is labelled by
what it became. Sector labels are used for sector-relative numbers only and
are never shown to a model, so they can't help it recognise a company.
"""

from __future__ import annotations

_GROUPS = {
    "banks": "HSBA LLOY BARC NWG STAN",
    "insurance": "PRU LGEN AV ADM BEZ HSX PHNX",
    "financial_services": "LSEG III SDR ICG MNG IGG RAT AJB PLUS EMG ASHM JUP QLT",
    "mining": "RIO AAL GLEN ANTO FRES",
    "oil_gas": "SHEL BP",
    "utilities": "NG SSE UU SVT CNA DRX PNN",
    "health": "AZN GSK HLN SN HIK GNS",
    "staples": "ULVR DGE BATS IMB TATE PFD BAG",
    "retail": "TSCO SBRY NXT MKS JD KGF BRBY BME FRAS GRG INCH WOSG OCDO",
    "travel_leisure": "CPG IHG WTB ENT IAG EZJ JET2 GAW DOM TRN",
    "housing_property": "BDEV PSN TW BKG BWY VTY CRST LAND BLND SGRO LMP DLN SAFE BYG UTG",
    "industrials": "BA RR AHT SMIN HLMA SPX BNZL HWDN DPLM IMI WEIR RTO DCC MRO BBY TPK PAGE QQ ROR RS1 "
                   "SXS SRP VSVS BAB BOY HILS MTO CKN HAS EXPN ITRK",
    "materials": "CRDA MNDI SMDS JMAT",
    "tech_media": "REL SGE AUTO RMV WPP INF PSON CCC ITV SPT FOUR RWS KNOS",
    "telecoms": "VOD HTWS",
}

SECTORS: dict[str, str] = {f"{t}.L": g for g, names in _GROUPS.items() for t in names.split()}

OTHER = "other"


def sector_of(symbol: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and symbol in overrides:
        return overrides[symbol]
    return SECTORS.get(symbol, OTHER)
