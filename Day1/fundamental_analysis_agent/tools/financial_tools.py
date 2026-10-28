"""Strands tools for retrieving and analyzing company financial statements."""

import math
from datetime import date, datetime
from typing import Any

import yfinance as yf
from strands import tool


def _number(value: Any) -> int | float | None:
    """Convert a provider value to a JSON-safe number."""
    if value is None:
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _period_label(period: Any) -> str:
    if isinstance(period, (date, datetime)):
        return period.date().isoformat() if isinstance(period, datetime) else period.isoformat()
    if hasattr(period, "date"):
        return period.date().isoformat()
    return str(period)


def _get_statement(symbol: str, attribute: str, label: str) -> dict[str, Any]:
    ticker = yf.Ticker(symbol)
    statement = getattr(ticker, attribute)
    if statement.empty:
        return {
            "error": f"No {label} data was returned for {symbol}.",
            "symbol": symbol,
        }

    periods = []
    for period in sorted(statement.columns, reverse=True):
        items = {
            str(item): _number(value)
            for item, value in statement[period].items()
        }
        periods.append({"period": _period_label(period), "items": items})

    return {
        "symbol": symbol,
        "statement": label,
        "period_basis": "annual",
        "periods": periods,
    }


def _row_value(statement: Any, row_names: tuple[str, ...], period: Any) -> int | float | None:
    available_rows = {str(row).casefold(): row for row in statement.index}
    for name in row_names:
        row = available_rows.get(name.casefold())
        if row is not None:
            value = _number(statement.at[row, period])
            if value is not None:
                return value
    return None


def _ratio(numerator: int | float | None, denominator: int | float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return round(numerator / denominator, 4)


@tool
def get_income_statment(symbol: str) -> dict[str, Any]:
    """Get a company's annual income statement from Yahoo Finance.

    Args:
        symbol: The stock ticker symbol, such as AAPL or MSFT.

    Returns:
        The annual income statement, with values grouped by fiscal period.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "A stock ticker symbol is required."}
    return _get_statement(symbol, "income_stmt", "income statement")


@tool
def get_balance_sheet(symbol: str) -> dict[str, Any]:
    """Get a company's annual balance sheet from Yahoo Finance.

    Args:
        symbol: The stock ticker symbol, such as AAPL or MSFT.

    Returns:
        The annual balance sheet, with values grouped by fiscal period.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "A stock ticker symbol is required."}
    return _get_statement(symbol, "balance_sheet", "balance sheet")


@tool
def get_cash_flow_statement(symbol: str) -> dict[str, Any]:
    """Get a company's annual cash flow statement from Yahoo Finance.

    Args:
        symbol: The stock ticker symbol, such as AAPL or MSFT.

    Returns:
        The annual cash flow statement, with values grouped by fiscal period.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "A stock ticker symbol is required."}
    return _get_statement(symbol, "cashflow", "cash flow statement")


@tool
def compute_key_ratios(symbol: str) -> dict[str, Any]:
    """Compute key profitability, liquidity, leverage, and return ratios.

    Ratios use the latest annual income statement and balance sheet available.
    Return-on-equity and return-on-assets use average balances when prior-year
    balances are available; otherwise they use the latest balance.

    Args:
        symbol: The stock ticker symbol, such as AAPL or MSFT.

    Returns:
        Ratios as decimals (for example, 0.25 means 25%); unavailable ratios
        are returned as null.
    """
    symbol = symbol.strip().upper()
    if not symbol:
        return {"error": "A stock ticker symbol is required."}

    ticker = yf.Ticker(symbol)
    income = ticker.income_stmt
    balance = ticker.balance_sheet
    cash_flow = ticker.cashflow

    if income.empty or balance.empty:
        missing = []
        if income.empty:
            missing.append("income statement")
        if balance.empty:
            missing.append("balance sheet")
        return {
            "error": f"Cannot compute ratios for {symbol}: no "
            f"{' or '.join(missing)} data was returned.",
            "symbol": symbol,
        }

    income_period = sorted(income.columns, reverse=True)[0]
    balance_periods = sorted(balance.columns, reverse=True)
    balance_period = balance_periods[0]

    revenue = _row_value(income, ("Total Revenue", "Operating Revenue"), income_period)
    gross_profit = _row_value(income, ("Gross Profit",), income_period)
    operating_income = _row_value(income, ("Operating Income",), income_period)
    net_income = _row_value(
        income,
        ("Net Income", "Net Income Common Stockholders"),
        income_period,
    )

    current_assets = _row_value(balance, ("Current Assets",), balance_period)
    current_liabilities = _row_value(balance, ("Current Liabilities",), balance_period)
    cash_and_investments = _row_value(
        balance,
        (
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents And Short Term Investments",
        ),
        balance_period,
    )
    if cash_and_investments is None:
        cash = _row_value(
            balance,
            ("Cash And Cash Equivalents", "Cash Cash Equivalents And Short Term Investments"),
            balance_period,
        )
        short_term_investments = _row_value(
            balance,
            ("Other Short Term Investments", "Short Term Investments"),
            balance_period,
        )
        if cash is not None or short_term_investments is not None:
            cash_and_investments = (cash or 0) + (short_term_investments or 0)
    receivables = _row_value(
        balance,
        ("Accounts Receivable", "Net Receivables", "Receivables"),
        balance_period,
    )
    quick_assets = (
        cash_and_investments + receivables
        if cash_and_investments is not None and receivables is not None
        else None
    )
    total_debt = _row_value(balance, ("Total Debt",), balance_period)
    if total_debt is None:
        current_debt = _row_value(balance, ("Current Debt", "Current Debt And Lease Obligation"), balance_period)
        long_term_debt = _row_value(balance, ("Long Term Debt", "Long Term Debt And Capital Lease Obligation"), balance_period)
        if current_debt is not None or long_term_debt is not None:
            total_debt = (current_debt or 0) + (long_term_debt or 0)

    equity = _row_value(
        balance,
        ("Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"),
        balance_period,
    )
    assets = _row_value(balance, ("Total Assets",), balance_period)

    prior_balance_period = balance_periods[1] if len(balance_periods) > 1 else None
    prior_equity = (
        _row_value(
            balance,
            ("Stockholders Equity", "Total Stockholder Equity", "Total Equity Gross Minority Interest"),
            prior_balance_period,
        )
        if prior_balance_period is not None
        else None
    )
    prior_assets = (
        _row_value(balance, ("Total Assets",), prior_balance_period)
        if prior_balance_period is not None
        else None
    )
    average_equity = (equity + prior_equity) / 2 if equity is not None and prior_equity is not None else equity
    average_assets = (assets + prior_assets) / 2 if assets is not None and prior_assets is not None else assets

    ratios: dict[str, float | None] = {
        "gross_margin": _ratio(gross_profit, revenue),
        "operating_margin": _ratio(operating_income, revenue),
        "net_profit_margin": _ratio(net_income, revenue),
        "current_ratio": _ratio(current_assets, current_liabilities),
        "quick_ratio": _ratio(quick_assets, current_liabilities),
        "debt_to_equity": _ratio(total_debt, equity),
        "return_on_equity": _ratio(net_income, average_equity),
        "return_on_assets": _ratio(net_income, average_assets),
    }

    if not cash_flow.empty:
        cash_flow_periods = sorted(cash_flow.columns, reverse=True)
        operating_cash_flow = _row_value(
            cash_flow,
            ("Operating Cash Flow", "Cash Flow From Continuing Operating Activities"),
            cash_flow_periods[0],
        )
        ratios["operating_cash_flow_to_net_income"] = _ratio(
            operating_cash_flow,
            net_income,
        )

    return {
        "symbol": symbol,
        "income_statement_period": _period_label(income_period),
        "balance_sheet_period": _period_label(balance_period),
        "ratios": ratios,
        "units": "ratios are decimals; multiply by 100 for percentages",
    }