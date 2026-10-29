"""Prices in real numbers: formatting, and an estimated cost per person.

The estimate is plain arithmetic on the menu, so it can be explained in one line:
  - take the priced food dishes on the translated menu (drinks left out);
  - the middle half of those prices is a "typical dish" (25th to 75th percentile);
  - a meal is about two dishes per person (e.g. a starter and a main),
  so the estimate is 2 x typical dish, shown as a range.
If no menu prices were found but a food guide quotes a price per person, that is used instead.
The estimate is then compared with the traveller's budget: fits, partly fits, or over budget.
"""
from __future__ import annotations

import re
import statistics
from typing import Optional

from .models import CostEstimate, MenuItem, UserQuery

SYMBOL_AFTER = {"EUR": "€", "CHF": "CHF", "SEK": "kr", "NOK": "kr", "DKK": "kr", "ISK": "kr", "CZK": "Kč",
                "PLN": "zł", "HUF": "Ft", "RON": "lei", "VND": "₫", "TRY": "₺", "AED": "AED", "SAR": "SAR",
                "QAR": "QAR", "MAD": "MAD", "UAH": "₴", "RUB": "₽"}
SYMBOL_BEFORE = {"GBP": "£", "USD": "$", "INR": "₹", "JPY": "¥", "CNY": "¥", "KRW": "₩", "AUD": "A$", "CAD": "C$",
                 "NZD": "NZ$", "SGD": "S$", "HKD": "HK$", "TWD": "NT$", "MXN": "MX$", "BRL": "R$", "THB": "฿",
                 "IDR": "Rp ", "MYR": "RM ", "PHP": "₱", "ILS": "₪", "ZAR": "R ", "EGP": "E£", "PKR": "Rs ",
                 "LKR": "Rs ", "NPR": "Rs ", "BDT": "৳", "NGN": "₦", "KES": "KSh ", "ARS": "AR$", "CLP": "CLP$",
                 "COP": "COL$", "PEN": "S/ "}
DRINK_WORDS = re.compile(r"drink|beverage|wine|beer|cocktail|coffee|tea\b|juice|soda|water|spirits|sake|soju|"
                         r"bebida|vino|cerveza|vins?\b|boisson|bevand|getränk|bier|cava|vermut|lassi|chai|"
                         r"飲み物|ドリンク|饮料|飲料|음료|เครื่องดื่ม|напитки|içecek|ποτά|napoje", re.I)
DISHES_PER_PERSON = 2


def _num(v: float) -> str:
    """12.5 -> '12.50', 7 -> '7', 1200 -> '1,200' (thousands separated, for ¥, ₩, Rp...)."""
    return f"{v:,.2f}" if v != int(v) else f"{int(v):,}"


def fmt_price(v, currency: str = "") -> str:
    """7 EUR -> '7 €', 12.5 GBP -> '£12.50'. Empty string when there is no price."""
    if v is None:
        return ""
    cur = (currency or "").upper()
    if cur in SYMBOL_BEFORE:
        return f"{SYMBOL_BEFORE[cur]}{_num(v)}"
    if cur in SYMBOL_AFTER:
        return f"{_num(v)} {SYMBOL_AFTER[cur]}"
    return f"{_num(v)} {currency}".strip()


def _round(v: float) -> int:
    return int(v + 0.5)       # 17.5 -> 18 (Python's round() would give 18 but 16.5 -> 16)


def fmt_range(lo, hi, currency: str = "", exact: bool = False) -> str:
    """(20, 30, EUR) -> '20–30 €'; (20, 20, GBP) -> '£20'. exact=True keeps cents: '8.50–11.50 €'."""
    if lo is None and hi is None:
        return ""
    conv = (lambda v: _num(round(v, 2))) if exact else (lambda v: f"{_round(v):,}")
    if lo is None or hi is None or conv(lo) == conv(hi):
        v = hi if hi is not None else lo
        return fmt_price(round(v, 2) if exact else _round(v), currency)
    cur = (currency or "").upper()
    a, b = conv(lo), conv(hi)
    if cur in SYMBOL_BEFORE:
        return f"{SYMBOL_BEFORE[cur]}{a}–{b}"
    sym = SYMBOL_AFTER.get(cur, currency)
    return f"{a}–{b} {sym}".strip()


def budget_text(q: UserQuery) -> str:
    """'up to 35 € per person' / 'about 15–25 € per person (my estimate for "cheap")' / 'no limit given'."""
    if q.budget_max is None and q.budget_min is None:
        return "no limit given"
    if q.budget_min is None:
        text = f"up to {fmt_price(q.budget_max, q.currency)} per person"
    elif q.budget_max is None:
        text = f"from {fmt_price(q.budget_min, q.currency)} per person"
    else:
        text = f"{fmt_range(q.budget_min, q.budget_max, q.currency)} per person"
    if q.budget_estimated:
        text += f' (my estimate for "{q.budget_note}")' if q.budget_note else " (estimated)"
    return text


def _percentile(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def judge(lo: float, hi: float, q: UserQuery) -> tuple[str, str]:
    """Compare a cost range with the budget -> (fit, sentence)."""
    cap, floor = q.budget_max, q.budget_min
    cur = q.currency
    if cap is None and floor is None:
        return "no_budget", "No budget was given."
    if cap is not None and lo > cap:
        return "over", f"Even the cheaper end ({fmt_price(_round(lo), cur)}) is above your {fmt_price(cap, cur)} limit."
    if cap is not None and hi > cap:
        return "partly", (f"Fits your {fmt_price(cap, cur)} limit if you choose the cheaper dishes; "
                          f"the pricier end reaches {fmt_price(_round(hi), cur)}.")
    if floor is not None and hi < floor * 0.7:
        return "under", f"Cheaper than the {fmt_price(floor, cur)} you were planning to spend."
    limit = f"your {fmt_price(cap, cur)} limit" if cap is not None else "your budget"
    return "fits", f"Fits {limit}."


def estimate_cost(items: list[MenuItem], q: UserQuery, guide_price: Optional[float] = None,
                  guide_currency: str = "") -> Optional[CostEstimate]:
    food = [it for it in items if it.price and it.price > 0 and not DRINK_WORDS.search(f"{it.section} {it.english_name}")]
    cur = next((it.currency for it in food if it.currency), "") or q.currency
    if len(food) >= 3:
        prices = [it.price for it in food]
        p25, p75, med = _percentile(prices, 0.25), _percentile(prices, 0.75), statistics.median(prices)
        lo, hi = DISHES_PER_PERSON * p25, DISHES_PER_PERSON * p75
        basis = (f"{len(food)} priced dishes on the menu. A typical dish costs {fmt_range(p25, p75, cur, exact=True)} "
                 f"(the middle half of the prices; median {fmt_price(round(med, 2), cur)}). "
                 f"A meal is about {DISHES_PER_PERSON} dishes per person, so about "
                 f"{fmt_range(lo, hi, cur)} per person, before drinks.")
        source = "menu"
    elif guide_price:
        cur = guide_currency or q.currency
        lo = hi = guide_price
        basis = f"No menu prices were found, so this uses the price a food guide quotes: about {fmt_price(guide_price, cur)} per person."
        source = "guide"
    else:
        return None
    fit, verdict = judge(lo, hi, q)
    return CostEstimate(low=round(lo, 2), high=round(hi, 2), currency=cur, source=source,
                        basis=basis, fit=fit, verdict=verdict)
