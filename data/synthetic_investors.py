"""
data/synthetic_investors.py
----------------------------
Generates synthetic CAMS investor data for demo/testing purposes.
Run this once to populate the vector database:

    python data/synthetic_investors.py

Creates 20 realistic investors with:
  - Personal details (name, PAN, folio, email, mobile)
  - Multiple mutual fund holdings per investor
  - SIP details, transaction history
  - KYC status
"""

from __future__ import annotations
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ── Fund Universe ─────────────────────────────────────────────────────────────
FUNDS = [
    {"name": "HDFC Top 100 Fund",             "category": "Large Cap",    "nav": 872.45},
    {"name": "SBI Bluechip Fund",             "category": "Large Cap",    "nav": 65.32},
    {"name": "Axis Midcap Fund",              "category": "Mid Cap",      "nav": 94.18},
    {"name": "Mirae Asset Emerging Bluechip", "category": "Large & Mid",  "nav": 112.60},
    {"name": "Parag Parikh Flexi Cap Fund",   "category": "Flexi Cap",    "nav": 68.75},
    {"name": "ICICI Pru Balanced Advantage",  "category": "Hybrid",       "nav": 55.40},
    {"name": "Kotak Small Cap Fund",          "category": "Small Cap",    "nav": 215.30},
    {"name": "Nippon India Small Cap",        "category": "Small Cap",    "nav": 148.90},
    {"name": "DSP Tax Saver Fund",            "category": "ELSS",         "nav": 88.15},
    {"name": "Aditya Birla Sun Life Liquid",  "category": "Liquid",       "nav": 382.10},
]

FIRST_NAMES = ["Arjun", "Priya", "Rahul", "Deepa", "Vikram", "Sunita", "Karthik",
               "Meena", "Arun", "Lakshmi", "Suresh", "Kavitha", "Rajesh", "Anitha",
               "Murugan", "Saranya", "Dinesh", "Padma", "Venkat", "Geetha"]
LAST_NAMES  = ["Kumar", "Sharma", "Patel", "Reddy", "Nair", "Iyer", "Pillai",
               "Singh", "Gupta", "Verma", "Rao", "Menon", "Krishnan", "Subramanian"]

CITIES      = ["Chennai", "Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Pune",
               "Kolkata", "Ahmedabad", "Coimbatore", "Kochi"]


def _random_pan() -> str:
    """Generate a realistic PAN number."""
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )


def _random_folio() -> str:
    return f"{random.randint(10000000, 99999999)}/{random.randint(10, 99)}"


def _random_mobile() -> str:
    return f"9{random.randint(100000000, 999999999)}"


def _random_date(years_back: int = 5) -> str:
    start = datetime.now() - timedelta(days=years_back * 365)
    random_days = random.randint(0, years_back * 365)
    return (start + timedelta(days=random_days)).strftime("%d-%m-%Y")


def _generate_holdings(num_funds: int = 3) -> list[dict]:
    """Generate random fund holdings for an investor."""
    selected = random.sample(FUNDS, num_funds)
    holdings = []
    for fund in selected:
        units     = round(random.uniform(50, 5000), 3)
        nav       = fund["nav"]
        value     = round(units * nav, 2)
        invested  = round(value * random.uniform(0.6, 1.2), 2)  # Some profit/loss
        gain_pct  = round((value - invested) / invested * 100, 2)
        has_sip   = random.choice([True, True, False])

        holding = {
            "fund_name":        fund["name"],
            "category":         fund["category"],
            "folio_number":     _random_folio(),
            "units":            units,
            "nav":              nav,
            "current_value":    value,
            "invested_amount":  invested,
            "gain_loss":        round(value - invested, 2),
            "gain_loss_pct":    gain_pct,
            "investment_date":  _random_date(years_back=5),
            "sip_active":       has_sip,
            "sip_amount":       random.choice([1000, 2000, 3000, 5000, 10000]) if has_sip else None,
            "sip_date":         random.randint(1, 28) if has_sip else None,  # SIP date of month
            "last_nav_date":    datetime.now().strftime("%d-%m-%Y"),
            "dividend_option":  random.choice(["Growth", "IDCW"]),
        }
        holdings.append(holding)
    return holdings


def _generate_recent_transactions(holdings: list[dict], num_txns: int = 3) -> list[dict]:
    """Generate recent transaction history."""
    txn_types = ["SIP", "Redemption", "Lump Sum Purchase", "Dividend Reinvestment"]
    txns      = []
    for _ in range(num_txns):
        fund   = random.choice(holdings)
        txn    = {
            "txn_id":    f"TXN{random.randint(1000000, 9999999)}",
            "fund_name": fund["fund_name"],
            "type":      random.choice(txn_types),
            "amount":    random.choice([1000, 2000, 5000, 10000, 25000, 50000]),
            "units":     round(random.uniform(5, 500), 3),
            "nav":       fund["nav"],
            "date":      _random_date(years_back=1),
            "status":    random.choice(["Processed", "Processed", "Processed", "Pending"]),
        }
        txns.append(txn)
    return txns


def generate_investors(count: int = 20) -> list[dict]:
    """Generate `count` synthetic investor records."""
    investors = []
    used_pans = set()

    for i in range(count):
        first = random.choice(FIRST_NAMES)
        last  = random.choice(LAST_NAMES)
        name  = f"{first} {last}"

        pan = _random_pan()
        while pan in used_pans:
            pan = _random_pan()
        used_pans.add(pan)

        num_funds = random.randint(2, 4)
        holdings  = _generate_holdings(num_funds)

        total_invested = sum(h["invested_amount"] for h in holdings)
        total_value    = sum(h["current_value"]   for h in holdings)
        total_gain     = round(total_value - total_invested, 2)
        total_gain_pct = round(total_gain / total_invested * 100, 2) if total_invested else 0

        active_sips = [h for h in holdings if h["sip_active"]]
        total_sip   = sum(h["sip_amount"] for h in active_sips)

        investor = {
            # ── Identity ──────────────────────────────────────────────
            "investor_id":   f"CAMS{str(i+1).zfill(5)}",
            "name":          name,
            "pan":           pan,
            "email":         f"{first.lower()}.{last.lower()}{random.randint(1,99)}@gmail.com",
            "mobile":        _random_mobile(),
            "city":          random.choice(CITIES),
            "dob":           _random_date(years_back=40),
            "kyc_status":    random.choice(["Verified", "Verified", "Verified", "Pending"]),
            "account_since": _random_date(years_back=7),

            # ── Portfolio Summary ──────────────────────────────────────
            "total_invested":     round(total_invested, 2),
            "total_current_value":round(total_value, 2),
            "total_gain_loss":    total_gain,
            "total_gain_loss_pct":total_gain_pct,
            "total_sip_amount":   total_sip,
            "num_funds":          num_funds,
            "holdings":           holdings,
            "recent_transactions":_generate_recent_transactions(holdings),

            # ── Nomination ─────────────────────────────────────────────
            "nominee_name":    f"{random.choice(FIRST_NAMES)} {last}",
            "nominee_relation":random.choice(["Spouse", "Son", "Daughter", "Father", "Mother"]),
        }
        investors.append(investor)

    return investors


if __name__ == "__main__":
    print("Generating synthetic investor data...")
    investors = generate_investors(20)

    # Save to JSON for reference
    out_path = Path(__file__).parent / "investors.json"
    with open(out_path, "w") as f:
        json.dump(investors, f, indent=2)

    print(f"✅ Generated {len(investors)} investors → {out_path}")
    print("\nSample investor:")
    sample = investors[0]
    print(f"  Name:          {sample['name']}")
    print(f"  PAN:           {sample['pan']}")
    print(f"  Total Value:   ₹{sample['total_current_value']:,.2f}")
    print(f"  Total Invested:₹{sample['total_invested']:,.2f}")
    print(f"  Gain/Loss:     ₹{sample['total_gain_loss']:,.2f} ({sample['total_gain_loss_pct']}%)")
    print(f"  Active SIPs:   ₹{sample['total_sip_amount']:,}/month")
    print(f"  Funds:         {[h['fund_name'] for h in sample['holdings']]}")