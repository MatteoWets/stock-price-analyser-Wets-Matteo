import yfinance as yf 
import yfinance as yf
import pandas as pd


pd.set_option('display.width', None)   # Prevent line wrapping
pd.set_option('display.max_columns', None)

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

trainingdata = yf.download(
    tech_tickers,
    start="2010-01-01",
    end="2021-01-01",
    group_by='ticker',
    auto_adjust=False,    # keep all columns
    actions=True,         # include dividends & splits
    threads=True
)
testdata = yf.download(
    tech_tickers,
    start="2021-01-01",
    end="2025-11-01",
    group_by='ticker',
    auto_adjust=False,
    actions=True,
    threads=True
)



# Or flatten the structure:
# E.g., pivot so you have Date × (Ticker × Field)
trainingflat = trainingdata.stack(level=0).rename_axis(('Date', 'Ticker')).reset_index()

# Save to CSV
trainingflat.to_csv("Data/trainingdata.csv", index=False)


# Or flatten the structure:
# E.g., pivot so you have Date × (Ticker × Field)
testflat = testdata.stack(level=0).rename_axis(('Date', 'Ticker')).reset_index()

# Save to CSV
testflat.to_csv("Data/testdata.csv", index=False)

