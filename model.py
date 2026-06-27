"""
Probability Model
Converts weather forecasts into calibrated probability estimates for Kalshi contracts.

The core idea:
- We pull forecasts from multiple sources (NWS, GFS, ECMWF) via the ensemble module
- The ensemble gives us a best-estimate high temp AND a measure of model disagreement
- We model the actual temperature as: actual ~ Normal(ensemble_mean, ensemble_std)
- From that distribution, we can compute P(temp > threshold) or P(temp in bracket)

When the ensemble is unavailable, we fall back to NWS-only with default error estimates.
"""

from dataclasses import dataclass
from scipy import stats

from weather_ensemble import build_ensemble
from weather import (
    CITIES,
    get_daily_high_forecast,
    get_forecast_lead_days,
    get_forecast_error_std,
    NWS_FORECAST_BIAS,
)


@dataclass
class ContractPrediction:
    """Model's prediction for a single Kalshi contract."""
    ticker: str
    contract_type: str        # 'threshold_above', 'threshold_below', 'bracket'
    description: str
    model_probability: float  # our estimated probability of YES
    forecast_high: float      # the NWS forecast we based this on
    error_std: float          # the uncertainty we're using
    threshold: float = None   # for threshold contracts
    bracket_low: float = None # for bracket contracts
    bracket_high: float = None


def parse_contract_ticker(ticker: str) -> dict:
    """
    Parse a Kalshi weather ticker into its components.

    Examples:
        KXHIGHCHI-26MAY05-T65  -> threshold above 65
        KXHIGHCHI-26MAY05-T58  -> threshold (check title for above/below)
        KXHIGHCHI-26MAY05-B64.5 -> bracket centered on 64.5 (means 64-65)

    The T contracts are "above" if the title says ">X" and "below" if "<X".
    The B contracts are brackets: B83.5 means 83-84, B81.5 means 81-82, etc.
    """
    parts = ticker.split("-")
    if len(parts) < 3:
        return {"type": "unknown", "ticker": ticker}

    series = parts[0]        # e.g. KXHIGHCHI
    date_str = parts[1]      # e.g. 26MAY05
    strike = parts[2]        # e.g. T65 or B64.5

    # Parse date: 26MAY05 -> 2026-05-05
    year = f"20{date_str[:2]}"
    month_str = date_str[2:5]
    day = date_str[5:]
    months = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
              "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
              "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}
    month = months.get(month_str, "01")
    date = f"{year}-{month}-{day}"

    if strike.startswith("T"):
        threshold = float(strike[1:])
        return {
            "type": "threshold",
            "series": series,
            "date": date,
            "threshold": threshold,
            "ticker": ticker,
        }
    elif strike.startswith("B"):
        center = float(strike[1:])
        # B83.5 means the bracket 83-84 (center ± 0.5, so range is center-0.5 to center+0.5)
        return {
            "type": "bracket",
            "series": series,
            "date": date,
            "bracket_low": center - 0.5,
            "bracket_high": center + 0.5,
            "center": center,
            "ticker": ticker,
        }
    else:
        return {"type": "unknown", "series": series, "date": date, "ticker": ticker}


def compute_probability(forecast_high: float, error_std: float, bias: float,
                        contract_info: dict, title: str = "") -> float:
    """
    Compute the probability of a contract paying out YES.

    We model: actual_temp ~ Normal(forecast_high + bias, error_std)

    For threshold contracts:
        - "above": P(actual > threshold)
        - "below": P(actual < threshold)

    For bracket contracts:
        - P(bracket_low <= actual <= bracket_high)

    Probabilities are clamped to [0.03, 0.97] because:
    - No weather forecast is ever 99%+ certain
    - The market has information we don't (local observers, newer data)
    - Overconfident probabilities create false "edge" that bleeds money
    """
    PROB_FLOOR = 0.03
    PROB_CEILING = 0.97

    # Our distribution of the actual high temperature
    dist = stats.norm(loc=forecast_high + bias, scale=error_std)

    if contract_info["type"] == "threshold":
        threshold = contract_info["threshold"]

        # Determine direction from title
        title_lower = title.lower()
        if ">" in title or "above" in title_lower:
            # P(temp > threshold)
            prob = 1 - dist.cdf(threshold)
        elif "<" in title or "below" in title_lower:
            # P(temp < threshold)
            prob = dist.cdf(threshold)
        else:
            # Default to "above" for T contracts
            prob = 1 - dist.cdf(threshold)

        return max(PROB_FLOOR, min(PROB_CEILING, prob))

    elif contract_info["type"] == "bracket":
        low = contract_info["bracket_low"]
        high = contract_info["bracket_high"]
        # Settlement is the integer (rounded) daily high. The bracket pays YES
        # if that rounded high is any integer in [low, high]. A continuous temp
        # rounds into that set when it falls in [low - 0.5, high + 0.5).
        # Using cdf(high) - cdf(low) here understated the bracket width by 1°F;
        # this matches the rounding logic used in sniper.py's verifier.
        prob = dist.cdf(high + 0.5) - dist.cdf(low - 0.5)
        # For brackets, floor is even lower since they're naturally low-probability
        return max(0.01, min(PROB_CEILING, prob))

    return 0.5  # fallback for unknown contract types


def predict_contract(ticker: str, title: str, city_key: str,
                     ensemble_cache: dict = None) -> ContractPrediction:
    """
    Generate a probability prediction for a single contract.

    ticker: Kalshi market ticker (e.g. 'KXHIGHCHI-26MAY05-T65')
    title: Market title (needed to determine above/below for threshold contracts)
    city_key: Key into CITIES dict ('chicago', 'nyc', 'miami')
    ensemble_cache: optional {date: EnsembleForecast} to avoid re-fetching
    """
    contract_info = parse_contract_ticker(ticker)
    city = CITIES[city_key]
    target_date = contract_info.get("date")

    # Try ensemble forecast first (multiple models combined)
    ensemble = None
    if ensemble_cache and target_date in ensemble_cache:
        ensemble = ensemble_cache[target_date]
    else:
        try:
            ensemble = build_ensemble(city_key, target_date)
            if ensemble_cache is not None:
                ensemble_cache[target_date] = ensemble
        except Exception as e:
            print(f"  Warning: ensemble failed, falling back to NWS-only: {e}")

    if ensemble and ensemble.ensemble_high_f is not None:
        # Use ensemble forecast
        forecast_high = ensemble.ensemble_high_f
        error_std = ensemble.ensemble_std
        bias = 0.0  # ensemble already accounts for model biases
    else:
        # Fall back to NWS-only
        forecast = get_daily_high_forecast(city, target_date)
        forecast_high = forecast["forecast_high_f"]
        if forecast_high is None:
            raise ValueError(f"No forecast available for {city.name} on {target_date}")

        lead_days = get_forecast_lead_days(target_date)
        error_std = get_forecast_error_std(lead_days)
        bias = NWS_FORECAST_BIAS.get(city_key, 0.0)

    # For same-day contracts, also check if peak has passed
    lead_days = get_forecast_lead_days(target_date)
    if lead_days <= 0:
        try:
            nws_forecast = get_daily_high_forecast(city, target_date)
            peak_confidence = nws_forecast.get("peak_confidence", "low")
            if peak_confidence == "high":
                # Peak is done — use observed max with tight uncertainty
                forecast_high = nws_forecast["forecast_high_f"]
                error_std = 1.0
                bias = 0.0
            elif peak_confidence == "medium":
                forecast_high = nws_forecast["forecast_high_f"]
                error_std = 1.5
                bias = 0.0
        except Exception:
            pass

    prob = compute_probability(forecast_high, error_std, bias, contract_info, title)

    # Build description
    if contract_info["type"] == "threshold":
        desc = f"P(high {'>' if '>' in title else '<'} {contract_info['threshold']}°F)"
    elif contract_info["type"] == "bracket":
        desc = f"P({contract_info['bracket_low']}° ≤ high ≤ {contract_info['bracket_high']}°F)"
    else:
        desc = "Unknown contract type"

    return ContractPrediction(
        ticker=ticker,
        contract_type=contract_info["type"],
        description=desc,
        model_probability=prob,
        forecast_high=forecast_high,
        error_std=error_std,
        threshold=contract_info.get("threshold"),
        bracket_low=contract_info.get("bracket_low"),
        bracket_high=contract_info.get("bracket_high"),
    )


def predict_all_for_city(city_key: str, markets: list[dict]) -> list[ContractPrediction]:
    """
    Generate predictions for all markets in a city's series.

    markets: list of market dicts from the Kalshi API
    """
    predictions = []
    city = CITIES[city_key]
    ensemble_cache = {}  # {date: EnsembleForecast} — avoids re-fetching per contract

    for m in markets:
        ticker = m.get("ticker", "")
        title = m.get("title", "")

        # Only process markets from this city's series
        if not ticker.startswith(city.kalshi_series):
            continue

        try:
            pred = predict_contract(ticker, title, city_key, ensemble_cache)
            predictions.append(pred)
        except Exception as e:
            print(f"  Warning: could not predict {ticker}: {e}")

    return predictions


def print_predictions(predictions: list[ContractPrediction]) -> None:
    """Pretty-print model predictions."""
    print(f"\n{'Ticker':<45} {'Description':<30} {'Model P':<10} {'Forecast'}")
    print("-" * 100)

    for p in sorted(predictions, key=lambda x: x.model_probability, reverse=True):
        print(f"  {p.ticker:<43} {p.description:<28} {p.model_probability:>6.1%}    "
              f"(high={p.forecast_high}°F ± {p.error_std}°)")


if __name__ == "__main__":
    import argparse
    import requests

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    parser = argparse.ArgumentParser(description="Generate predictions for weather contracts")
    parser.add_argument("city", choices=list(CITIES.keys()), help="City to predict")
    args = parser.parse_args()

    city = CITIES[args.city]

    # Fetch current markets for this series
    print(f"Fetching markets for {city.name} ({city.kalshi_series})...")
    resp = requests.get(f"{BASE_URL}/markets", params={
        "limit": 200,
        "status": "open",
        "series_ticker": city.kalshi_series,
    })
    resp.raise_for_status()
    markets = resp.json().get("markets", [])
    print(f"Found {len(markets)} markets")

    # Generate predictions
    print(f"\nFetching NWS forecast for {city.name}...")
    predictions = predict_all_for_city(args.city, markets)

    print_predictions(predictions)
