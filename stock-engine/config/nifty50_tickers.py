"""NIFTY 50 ticker universe with company names and search keywords.

Each entry maps a Yahoo Finance ticker (with .NS suffix) to:
  - company : official company name
  - sector  : GICS-style sector classification
  - keywords: 2-3 search terms used to match GDELT articles to this stock
  - cap_tier: approximate market-cap tier as of 2024
              "TOP10" (rank 1-10), "MID" (11-25), "TAIL" (26-50)

Maintained manually — update when NIFTY 50 index reconstitution occurs.
"""

NIFTY50_TICKERS: dict[str, dict] = {
    # ------- TOP 10 by market cap (approx 2024) -------
    "RELIANCE.NS": {
        "company": "Reliance Industries Ltd",
        "sector": "Energy",
        "keywords": ["Reliance Industries", "RIL", "Mukesh Ambani"],
        "cap_tier": "TOP10",
    },
    "TCS.NS": {
        "company": "Tata Consultancy Services Ltd",
        "sector": "Information Technology",
        "keywords": ["TCS", "Tata Consultancy", "Tata IT"],
        "cap_tier": "TOP10",
    },
    "HDFCBANK.NS": {
        "company": "HDFC Bank Ltd",
        "sector": "Financials",
        "keywords": ["HDFC Bank", "HDFC", "Sashidhar Jagdishan"],
        "cap_tier": "TOP10",
    },
    "INFY.NS": {
        "company": "Infosys Ltd",
        "sector": "Information Technology",
        "keywords": ["Infosys", "INFY", "Salil Parekh"],
        "cap_tier": "TOP10",
    },
    "ICICIBANK.NS": {
        "company": "ICICI Bank Ltd",
        "sector": "Financials",
        "keywords": ["ICICI Bank", "ICICI", "Sandeep Bakhshi"],
        "cap_tier": "TOP10",
    },
    "BHARTIARTL.NS": {
        "company": "Bharti Airtel Ltd",
        "sector": "Communication Services",
        "keywords": ["Bharti Airtel", "Airtel", "Sunil Mittal"],
        "cap_tier": "TOP10",
    },
    "SBIN.NS": {
        "company": "State Bank of India",
        "sector": "Financials",
        "keywords": ["SBI", "State Bank of India", "State Bank"],
        "cap_tier": "TOP10",
    },
    "ITC.NS": {
        "company": "ITC Ltd",
        "sector": "Consumer Staples",
        "keywords": ["ITC", "ITC Limited", "Sanjiv Puri"],
        "cap_tier": "TOP10",
    },
    "HINDUNILVR.NS": {
        "company": "Hindustan Unilever Ltd",
        "sector": "Consumer Staples",
        "keywords": ["Hindustan Unilever", "HUL", "Unilever India"],
        "cap_tier": "TOP10",
    },
    "LT.NS": {
        "company": "Larsen & Toubro Ltd",
        "sector": "Industrials",
        "keywords": ["Larsen Toubro", "L&T", "LT Infrastructure"],
        "cap_tier": "TOP10",
    },

    # ------- MID tier (rank 11-25) -------
    "BAJFINANCE.NS": {
        "company": "Bajaj Finance Ltd",
        "sector": "Financials",
        "keywords": ["Bajaj Finance", "Bajaj Finserv", "Bajaj NBFC"],
        "cap_tier": "MID",
    },
    "KOTAKBANK.NS": {
        "company": "Kotak Mahindra Bank Ltd",
        "sector": "Financials",
        "keywords": ["Kotak Mahindra Bank", "Kotak Bank", "Uday Kotak"],
        "cap_tier": "MID",
    },
    "HCLTECH.NS": {
        "company": "HCL Technologies Ltd",
        "sector": "Information Technology",
        "keywords": ["HCL Technologies", "HCL Tech", "HCLTech"],
        "cap_tier": "MID",
    },
    "AXISBANK.NS": {
        "company": "Axis Bank Ltd",
        "sector": "Financials",
        "keywords": ["Axis Bank", "Axis", "Amitabh Chaudhry"],
        "cap_tier": "MID",
    },
    "TITAN.NS": {
        "company": "Titan Company Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Titan Company", "Titan", "Tanishq"],
        "cap_tier": "MID",
    },
    "ASIANPAINT.NS": {
        "company": "Asian Paints Ltd",
        "sector": "Materials",
        "keywords": ["Asian Paints", "AsianPaints", "Asian Paint"],
        "cap_tier": "MID",
    },
    "MARUTI.NS": {
        "company": "Maruti Suzuki India Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Maruti Suzuki", "Maruti", "Suzuki India"],
        "cap_tier": "MID",
    },
    "SUNPHARMA.NS": {
        "company": "Sun Pharmaceutical Industries Ltd",
        "sector": "Healthcare",
        "keywords": ["Sun Pharma", "Sun Pharmaceutical", "Dilip Shanghvi"],
        "cap_tier": "MID",
    },
    "WIPRO.NS": {
        "company": "Wipro Ltd",
        "sector": "Information Technology",
        "keywords": ["Wipro", "Wipro IT", "Thierry Delaporte"],
        "cap_tier": "MID",
    },
    "ULTRACEMCO.NS": {
        "company": "UltraTech Cement Ltd",
        "sector": "Materials",
        "keywords": ["UltraTech Cement", "UltraTech", "Aditya Birla Cement"],
        "cap_tier": "MID",
    },
    "BAJAJFINSV.NS": {
        "company": "Bajaj Finserv Ltd",
        "sector": "Financials",
        "keywords": ["Bajaj Finserv", "Bajaj Financial Services", "Bajaj Holdings"],
        "cap_tier": "MID",
    },
    "ONGC.NS": {
        "company": "Oil and Natural Gas Corporation Ltd",
        "sector": "Energy",
        "keywords": ["ONGC", "Oil Natural Gas Corporation", "Oil India"],
        "cap_tier": "MID",
    },
    "NTPC.NS": {
        "company": "NTPC Ltd",
        "sector": "Utilities",
        "keywords": ["NTPC", "National Thermal Power", "NTPC Power"],
        "cap_tier": "MID",
    },
    "TATAMOTORS.NS": {
        "company": "Tata Motors Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Tata Motors", "Tata Auto", "Jaguar Land Rover"],
        "cap_tier": "MID",
        "note": "Delisted on Yahoo Finance as of 2026; data unavailable",
    },
    "POWERGRID.NS": {
        "company": "Power Grid Corporation of India Ltd",
        "sector": "Utilities",
        "keywords": ["Power Grid", "PowerGrid Corporation", "PGCIL"],
        "cap_tier": "MID",
    },

    # ------- TAIL tier (rank 26-50) -------
    "M&M.NS": {
        "company": "Mahindra & Mahindra Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Mahindra Mahindra", "M&M", "Anand Mahindra"],
        "cap_tier": "TAIL",
    },
    "JSWSTEEL.NS": {
        "company": "JSW Steel Ltd",
        "sector": "Materials",
        "keywords": ["JSW Steel", "JSW", "Sajjan Jindal"],
        "cap_tier": "TAIL",
    },
    "ADANIENT.NS": {
        "company": "Adani Enterprises Ltd",
        "sector": "Industrials",
        "keywords": ["Adani Enterprises", "Adani", "Gautam Adani"],
        "cap_tier": "TAIL",
    },
    "ADANIPORTS.NS": {
        "company": "Adani Ports and Special Economic Zone Ltd",
        "sector": "Industrials",
        "keywords": ["Adani Ports", "Adani Port", "APSEZ"],
        "cap_tier": "TAIL",
    },
    "TATASTEEL.NS": {
        "company": "Tata Steel Ltd",
        "sector": "Materials",
        "keywords": ["Tata Steel", "Tata Steelium", "TV Narendran"],
        "cap_tier": "TAIL",
    },
    "NESTLEIND.NS": {
        "company": "Nestle India Ltd",
        "sector": "Consumer Staples",
        "keywords": ["Nestle India", "Maggi", "Nestle"],
        "cap_tier": "TAIL",
    },
    "TECHM.NS": {
        "company": "Tech Mahindra Ltd",
        "sector": "Information Technology",
        "keywords": ["Tech Mahindra", "TechM", "Mohit Joshi"],
        "cap_tier": "TAIL",
    },
    "COALINDIA.NS": {
        "company": "Coal India Ltd",
        "sector": "Materials",
        "keywords": ["Coal India", "CIL", "Coal India Limited"],
        "cap_tier": "TAIL",
    },
    "HDFCLIFE.NS": {
        "company": "HDFC Life Insurance Company Ltd",
        "sector": "Financials",
        "keywords": ["HDFC Life", "HDFC Life Insurance", "HDFC Insurance"],
        "cap_tier": "TAIL",
    },
    "SBILIFE.NS": {
        "company": "SBI Life Insurance Company Ltd",
        "sector": "Financials",
        "keywords": ["SBI Life", "SBI Life Insurance", "SBI Insurance"],
        "cap_tier": "TAIL",
    },
    "BRITANNIA.NS": {
        "company": "Britannia Industries Ltd",
        "sector": "Consumer Staples",
        "keywords": ["Britannia", "Britannia Industries", "Good Day"],
        "cap_tier": "TAIL",
    },
    "GRASIM.NS": {
        "company": "Grasim Industries Ltd",
        "sector": "Materials",
        "keywords": ["Grasim Industries", "Grasim", "Aditya Birla Group"],
        "cap_tier": "TAIL",
    },
    "INDUSINDBK.NS": {
        "company": "IndusInd Bank Ltd",
        "sector": "Financials",
        "keywords": ["IndusInd Bank", "IndusInd", "Hindujas Bank"],
        "cap_tier": "TAIL",
    },
    "CIPLA.NS": {
        "company": "Cipla Ltd",
        "sector": "Healthcare",
        "keywords": ["Cipla", "Cipla Pharma", "Umang Vohra"],
        "cap_tier": "TAIL",
    },
    "HEROMOTOCO.NS": {
        "company": "Hero MotoCorp Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Hero MotoCorp", "Hero Honda", "Pawan Munjal"],
        "cap_tier": "TAIL",
    },
    "EICHERMOT.NS": {
        "company": "Eicher Motors Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Eicher Motors", "Royal Enfield", "Eicher"],
        "cap_tier": "TAIL",
    },
    "DIVISLAB.NS": {
        "company": "Divi's Laboratories Ltd",
        "sector": "Healthcare",
        "keywords": ["Divis Laboratories", "Divis Lab", "Divi's Lab"],
        "cap_tier": "TAIL",
    },
    "DRREDDY.NS": {
        "company": "Dr. Reddy's Laboratories Ltd",
        "sector": "Healthcare",
        "keywords": ["Dr Reddy", "Dr Reddys", "DRL"],
        "cap_tier": "TAIL",
    },
    "APOLLOHOSP.NS": {
        "company": "Apollo Hospitals Enterprise Ltd",
        "sector": "Healthcare",
        "keywords": ["Apollo Hospitals", "Apollo Hospital", "Prathap Reddy"],
        "cap_tier": "TAIL",
    },
    "TATACONSUM.NS": {
        "company": "Tata Consumer Products Ltd",
        "sector": "Consumer Staples",
        "keywords": ["Tata Consumer", "Tata Consumer Products", "Tata Tea"],
        "cap_tier": "TAIL",
    },
    "BPCL.NS": {
        "company": "Bharat Petroleum Corporation Ltd",
        "sector": "Energy",
        "keywords": ["BPCL", "Bharat Petroleum", "Bharat Petrol"],
        "cap_tier": "TAIL",
    },
    "HINDALCO.NS": {
        "company": "Hindalco Industries Ltd",
        "sector": "Materials",
        "keywords": ["Hindalco", "Hindalco Industries", "Novelis"],
        "cap_tier": "TAIL",
    },
    "WIPRO.NS_DUP": {   # placeholder — Wipro already in MID; keep 50 distinct
        "company": None,  # sentinel — see note below
        "sector": None,
        "keywords": [],
        "cap_tier": None,
    },
    "BAJAJ-AUTO.NS": {
        "company": "Bajaj Auto Ltd",
        "sector": "Consumer Discretionary",
        "keywords": ["Bajaj Auto", "Bajaj Motorcycle", "Rajiv Bajaj"],
        "cap_tier": "TAIL",
    },
    "SHRIRAMFIN.NS": {
        "company": "Shriram Finance Ltd",
        "sector": "Financials",
        "keywords": ["Shriram Finance", "Shriram Transport", "Shriram"],
        "cap_tier": "TAIL",
    },
}

# ------------------------------------------------------------------
# Remove the Wipro duplicate sentinel and replace with the 50th stock
# (composition changes; BEL was added in Sept 2024 reconstitution)
# ------------------------------------------------------------------
NIFTY50_TICKERS.pop("WIPRO.NS_DUP", None)
NIFTY50_TICKERS["BEL.NS"] = {
    "company": "Bharat Electronics Ltd",
    "sector": "Industrials",
    "keywords": ["Bharat Electronics", "BEL", "BEL Defence"],
    "cap_tier": "TAIL",
}

# ------------------------------------------------------------------
# Convenience views
# ------------------------------------------------------------------
def get_ticker_list() -> list[str]:
    """Return sorted list of Yahoo Finance ticker strings."""
    return sorted(NIFTY50_TICKERS.keys())


def get_keywords_map() -> dict[str, list[str]]:
    """Return {ticker: [keywords]} for GDELT matching."""
    return {t: info["keywords"] for t, info in NIFTY50_TICKERS.items()}


def get_sector_map() -> dict[str, str]:
    """Return {ticker: sector}."""
    return {t: info["sector"] for t, info in NIFTY50_TICKERS.items()}


def get_cap_tier_map() -> dict[str, str]:
    """Return {ticker: cap_tier}."""
    return {t: info["cap_tier"] for t, info in NIFTY50_TICKERS.items()}
