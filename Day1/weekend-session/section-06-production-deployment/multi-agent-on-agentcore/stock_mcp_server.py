"""
Stock MCP Server — exposes stock price lookup over the MCP protocol.

Uses yfinance to fetch live stock prices.

Local:  python stock_mcp_server.py          (http://localhost:8000/mcp)
Deploy: agentcore configure -e stock_mcp_server.py --protocol MCP ...
        (AgentCore expects MCP servers on port 8000, path /mcp — the defaults here)
"""

import yfinance as yf
from mcp.server.fastmcp import FastMCP

# stateless_http=True: every request is independent — required on AgentCore
mcp = FastMCP("StockTools", host="0.0.0.0", stateless_http=True)


@mcp.tool()
def get_stock_price(ticker: str) -> str:
    """Get the latest stock price for a given ticker symbol (e.g. AAPL, MSFT, AMZN)."""
    try:
        stock = yf.Ticker(ticker.strip().upper())
        info = stock.info
        price = info.get("currentPrice") or info.get("regularMarketPrice")
        name = info.get("shortName", ticker.upper())
        currency = info.get("currency", "USD")
        change = info.get("regularMarketChangePercent")

        if price is None:
            return f"Could not find price for '{ticker}'. Check the ticker symbol."

        result = f"{name} ({ticker.upper()}): {currency} {price:.2f}"
        if change is not None:
            direction = "▲" if change >= 0 else "▼"
            result += f" {direction} {abs(change):.2f}%"
        return result
    except Exception as e:
        return f"Error fetching {ticker}: {e}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
