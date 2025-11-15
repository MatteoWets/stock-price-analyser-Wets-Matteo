import yfinance as yf
import pandas as pd

pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Tech stock tickers
tech_tickers = [
    # Big Tech
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Alphabet
    "AMZN",  # Amazon
    "META",  # Meta Platforms
    "NVDA",  # Nvidia
    # Semiconductors
    "AMD",   # AMD
    "AVGO",  # Broadcom
    "INTC",  # Intel
    "QCOM",  # Qualcomm
    "TSM",   # Taiwan Semi
    "MU",    # Micron
    # Software & Cloud
    "ADBE",  # Adobe
    "CRM",   # Salesforce
    "ORCL",  # Oracle
    "SAP",   # SAP
    "NOW",   # ServiceNow
    "TEAM",  # Atlassian
    # Cybersecurity
    "PANW",  # Palo Alto Networks
    "CRWD",  # CrowdStrike
    "FTNT",  # Fortinet
    "OKTA",  # Okta
    # Networking & Hardware
    "CSCO",  # Cisco
    "DELL",  # Dell Technologies
    "HPE",   # HP Enterprise
    "HPQ",   # HP Inc.
    # Electronic equipment / chips
    "ADI",   # Analog Devices
    "TXN",   # Texas Instruments
    "NXPI",  # NXP Semiconductors
    "MRVL",  # Marvell Technology
]

# Market indices for feature engineering
market_tickers = [
    "^IXIC",  # Nasdaq Composite
    "^GSPC",  # S&P 500
    "SPY",    # S&P 500 ETF
    "^VIX",   # Volatility Index
]

print("=" * 70)
print("DOWNLOADING ALL DATA FROM YFINANCE")
print("=" * 70)

# Download all tech stocks (2010-2025)
print("\n1. Downloading tech stocks data (2010-2025)...")
print(f"   Tickers: {len(tech_tickers)} stocks")

tech_data = yf.download(
    tech_tickers,
    start="2010-01-01",
    end="2025-11-15",
    group_by='ticker',
    auto_adjust=False,
    actions=True,
    threads=True
)

# Flatten the multi-level structure
print("   Flattening tech stock data...")
tech_flat = tech_data.stack(level=0).rename_axis(('Date', 'Ticker')).reset_index()

# Download market indices (2010-2025)
print("\n2. Downloading market indices (Nasdaq, S&P 500, VIX)...")
print(f"   Tickers: {market_tickers}")

market_data_frames = []

for ticker in market_tickers:
    print(f"   - Downloading {ticker}...")
    data = yf.download(
        ticker,
        start="2010-01-01",
        end="2025-11-15",
        auto_adjust=True,
        progress=False
    )
    
    # Handle multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    # Reset index to make Date a column
    data = data.reset_index()
    
    # Add ticker column
    data['Ticker'] = ticker
    
    market_data_frames.append(data)

# Combine all market data
market_flat = pd.concat(market_data_frames, ignore_index=True)

# Combine tech stocks and market indices
print("\n3. Combining all data...")
all_data = pd.concat([tech_flat, market_flat], ignore_index=True)

# Sort by Date and Ticker
all_data = all_data.sort_values(['Date', 'Ticker']).reset_index(drop=True)

# Save to CSV
output_file = "data/raw.csv"
print(f"\n4. Saving to {output_file}...")
all_data.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE!")
print("=" * 70)
print(f"Total records: {len(all_data):,}")
print(f"Date range: {all_data['Date'].min()} to {all_data['Date'].max()}")
print(f"Unique tickers: {all_data['Ticker'].nunique()}")
print(f"\nTickers included:")
print(f"  - Tech stocks: {len(tech_tickers)}")
print(f"  - Market indices: {len(market_tickers)}")
print(f"\nFile saved: {output_file}")

# Display sample data
print("\n" + "=" * 70)
print("SAMPLE DATA (first 10 rows)")
print("=" * 70)
print(all_data.head(10))

print("\n" + "=" * 70)
print("DATA COLUMNS")
print("=" * 70)
print(all_data.columns.tolist())

print("\n" + "=" * 70)
print("DATA INFO")
print("=" * 70)
print(all_data.info())
