"""
DS528 Final Project — Synthetic Data Generator with Real-World Elements

Generates a 50,000-row synthetic fan dataset for the 2026 FIFA World Cup
travel demand prediction project.

Real-world elements injected:
  - 16 official 2026 World Cup host cities (USA, Canada, Mexico) with coordinates
  - ~30 real countries with actual GDP per capita, population, and region
  - Real haversine distances from each country centroid to nearest host city
  - Real US/Canada/Mexico visa requirements by nationality
  - Approximate FIFA rankings (as of mid-2025)
  - Realistic travel probability function using logit model

Output: data/synthetic_worldcup_fans.csv
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# SEED
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
rng = np.random.default_rng(RANDOM_STATE)

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = DATA_DIR / "synthetic_worldcup_fans.csv"

# ===========================================================================
# 1. REAL HOST CITIES (2026 FIFA World Cup)
# ===========================================================================
HOST_CITIES = [
    # USA (11)
    ("Atlanta",         33.7490,  -84.3880, "USA"),
    ("Boston",          42.3601,  -71.0589, "USA"),
    ("Dallas",          32.7767,  -96.7970, "USA"),
    ("Houston",         29.7604,  -95.3698, "USA"),
    ("Kansas City",     39.0997,  -94.5786, "USA"),
    ("Los Angeles",     34.0522, -118.2437, "USA"),
    ("Miami",           25.7617,  -80.1918, "USA"),
    ("New York",        40.7128,  -74.0060, "USA"),
    ("Philadelphia",    39.9526,  -75.1652, "USA"),
    ("San Francisco",   37.7749, -122.4194, "USA"),
    ("Seattle",         47.6062, -122.3321, "USA"),
    # Canada (2)
    ("Toronto",         43.6532,  -79.3832, "Canada"),
    ("Vancouver",       49.2827, -123.1207, "Canada"),
    # Mexico (3)
    ("Guadalajara",     20.6597, -103.3496, "Mexico"),
    ("Mexico City",     19.4326,  -99.1332, "Mexico"),
    ("Monterrey",       25.6866, -100.3161, "Mexico"),
]

# ===========================================================================
# 2. REAL COUNTRIES
# ===========================================================================
# (name, region, lat, lon, gdp_per_capita_usd, population_m, fifa_rank,
#  qualified_2026_prob, us_visa_required, canada_visa_required, mexico_visa_required)
COUNTRIES = [
    # ---- North America (hosts) ----
    ("United States",    "North America", 39.83,  -98.58,  76330, 332,  16, 1.00, False, False, False),
    ("Canada",           "North America", 56.13, -106.35,  52500,  38,  41, 1.00, False, False, False),
    ("Mexico",           "North America", 23.63, -102.55,  10700, 129,  14, 1.00, False, False, False),
    # ---- South America ----
    ("Brazil",           "South America",-14.24, -51.93,   8900, 214,   3, 1.00, True,  True,  False),
    ("Argentina",        "South America",-38.42, -63.62,  11000,  46,   1, 1.00, True,  True,  False),
    ("Colombia",         "South America",  4.57, -74.30,   6100,  52,  17, 0.80, True,  True,  False),
    ("Uruguay",          "South America",-32.52, -55.77,  17000,   3.5, 15, 0.70, True,  True,  False),
    ("Chile",            "South America",-35.68, -71.54,  15000,  19,  32, 0.55, False, False, False),
    ("Peru",             "South America", -9.19, -75.02,   6600,  34,  42, 0.40, True,  True,  False),
    ("Ecuador",          "South America", -1.83, -78.18,   6000,  18,  36, 0.50, True,  True,  False),
    # ---- Europe ----
    ("England",          "Europe",        52.36,  -1.17,  46500,  56,   5, 1.00, False, False, False),
    ("Germany",          "Europe",        51.17,  10.45,  51500,  83,  11, 1.00, False, False, False),
    ("France",           "Europe",        46.23,   2.21,  44500,  68,   2, 1.00, False, False, False),
    ("Spain",            "Europe",        40.46,  -3.75,  31000,  47,   8, 1.00, False, False, False),
    ("Italy",            "Europe",        41.87,  12.57,  36500,  59,  10, 0.90, False, False, False),
    ("Netherlands",      "Europe",        52.13,   5.29,  58000,  17.5,  6, 1.00, False, False, False),
    ("Portugal",         "Europe",        39.40,  -8.22,  25000,  10.3,  7, 1.00, False, False, False),
    ("Belgium",          "Europe",        50.50,   4.47,  52000,  11.6,  4, 1.00, False, False, False),
    ("Croatia",          "Europe",        45.10,  15.20,  18500,   3.9, 12, 0.80, False, False, False),
    ("Sweden",           "Europe",        60.13,  18.64,  60000,  10.5, 27, 0.60, False, False, False),
    ("Switzerland",      "Europe",        46.82,   8.23,  93500,   8.7, 19, 0.70, False, False, False),
    ("Turkey",           "Europe",        38.96,  35.24,   9800,  85,  38, 0.50, True,  True,  False),
    # ---- Asia ----
    ("Japan",            "Asia",          36.20, 138.25,  40000, 126,  20, 0.90, False, False, False),
    ("South Korea",      "Asia",          35.91, 127.77,  35000,  52,  24, 0.85, False, False, False),
    ("China",            "Asia",          35.86, 104.20,  12500,1412,  79, 0.30, True,  True,  False),
    ("India",            "Asia",          20.59,  78.96,   2400,1408, 120, 0.10, True,  True,  True),
    ("Saudi Arabia",     "Asia",          23.89,  45.08,  23000,  35,  53, 0.60, True,  True,  False),
    # ---- Africa ----
    ("Nigeria",          "Africa",         9.08,   8.68,   2100, 213,  30, 0.40, True,  True,  True),
    ("Egypt",            "Africa",        26.82,  30.80,   4000, 104,  35, 0.40, True,  True,  True),
    ("Morocco",          "Africa",        31.79,  -7.09,   3800,  37,  13, 0.90, True,  True,  False),
    ("Senegal",          "Africa",        14.50, -14.45,   1600,  17,  18, 0.70, True,  True,  True),
    ("Ghana",            "Africa",         7.95,  -1.02,   2500,  33,  60, 0.40, True,  True,  True),
    # ---- Oceania ----
    ("Australia",        "Oceania",      -25.27, 133.78,  62000,  26,  25, 0.75, False, False, False),
]

# ===========================================================================
# 3. MATCH TYPES
# ===========================================================================
MATCH_TYPES = [
    ("Group",         0.48),
    ("Round of 32",   0.17),
    ("Round of 16",   0.13),
    ("Quarter-final", 0.10),
    ("Semi-final",    0.07),
    ("Final",         0.05),
]

# ===========================================================================
# 4. HELPER: HAVERSINE DISTANCE
# ===========================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two (lat, lon) pairs."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_host_distance(country_lat, country_lon):
    """Distance from a country centroid to the nearest 2026 host city."""
    return min(
        haversine_km(country_lat, country_lon, h_lat, h_lon)
        for _, h_lat, h_lon, _ in HOST_CITIES
    )


# ===========================================================================
# 5. DATA GENERATION
# ===========================================================================
def generate_dataset(n_fans: int = 50_000) -> pd.DataFrame:
    """
    Generate the full synthetic dataset with real-world elements.
    """
    records = []

    # Pre-compute per-country distances and host info
    country_data = {}
    for c in COUNTRIES:
        name, region, lat, lon, gdp, pop, rank, qual, us_v, ca_v, mx_v = c
        dist = nearest_host_distance(lat, lon)
        # Nearest host country
        host_dists = []
        for h_name, h_lat, h_lon, h_country in HOST_CITIES:
            d = haversine_km(lat, lon, h_lat, h_lon)
            host_dists.append((d, h_name, h_country))
        host_dists.sort()
        nearest_host = host_dists[0]
        country_data[name] = {
            "region": region,
            "lat": lat, "lon": lon,
            "gdp": gdp, "pop": pop,
            "fifa_rank": rank,
            "qual_prob": qual,
            "us_visa": us_v, "ca_visa": ca_v, "mx_visa": mx_v,
            "distance_km": dist,
            "nearest_host_city": nearest_host[1],
            "nearest_host_country": nearest_host[2],
        }

    country_names = list(country_data.keys())
    # Weight country selection by population so larger countries appear more often
    country_weights = np.array([country_data[n]["pop"] for n in country_names], dtype=float)

    # Income levels based on GDP per capita
    def gdp_to_income(gdp):
        if gdp < 5000:
            return "Low"
        elif gdp < 20000:
            return "Medium"
        else:
            return "High"

    # Determine visa requirement for a fan's nearest host country
    def visa_needed(country_name, host_country):
        cd = country_data[country_name]
        if host_country == "USA":
            return cd["us_visa"]
        elif host_country == "Canada":
            return cd["ca_visa"]
        else:  # Mexico
            return cd["mx_visa"]

    fan_id = 0
    batch_size = 5000
    for batch_start in range(0, n_fans, batch_size):
        batch_end = min(batch_start + batch_size, n_fans)
        n_batch = batch_end - batch_start

        # Sample countries for this batch
        country_idx = rng.choice(len(country_names), size=n_batch, p=country_weights / country_weights.sum())
        batch_countries = [country_names[i] for i in country_idx]

        for j in range(n_batch):
            fan_id += 1
            cname = batch_countries[j]
            cd = country_data[cname]

            # --- Demographics ---
            age = int(rng.integers(18, 80))
            region = cd["region"]
            income_level = gdp_to_income(cd["gdp"])

            # --- Distance & host ---
            distance_km = round(cd["distance_km"] + rng.normal(0, cd["distance_km"] * 0.08), 1)
            distance_km = max(distance_km, 50)
            host_city = cd["nearest_host_city"]
            host_country = cd["nearest_host_country"]

            # --- Football profile ---
            fifa_rank = cd["fifa_rank"]
            # Teams with better FIFA rank have higher "qualified" probability (hosts = 1.0)
            team_qual_prob = cd["qual_prob"]
            favorite_team_qualified = int(rng.random() < team_qual_prob)

            # Engagement scores: influenced by country football culture, income, team qualification
            # Strong football nations = higher base engagement
            base_engagement = max(20, 90 - fifa_rank * 0.3)  # Better rank → higher engagement baseline
            base_engagement += rng.normal(0, 15)
            football_engagement = np.clip(base_engagement, 0, 100)
            football_engagement = round(football_engagement, 1)

            social_media = np.clip(football_engagement + rng.normal(0, 20), 0, 100)
            social_media = round(social_media, 1)

            # Previous attendance: correlated with age, income, engagement
            prev_attendance_base = (
                0.05
                + (age - 18) / 62 * 0.08
                + (football_engagement / 100) * 0.12
                + (2 if cd["gdp"] > 30000 else 1 if cd["gdp"] > 10000 else 0) * 0.06
            )
            previous_attendance = int(rng.random() < prev_attendance_base)

            # --- Search behavior ---
            # Correlated with engagement, time to match, match importance (assigned later)
            engagement_factor = football_engagement / 100
            ticket_search_count = rng.poisson(0.5 + engagement_factor * 3)
            flight_search_count = rng.poisson(0.3 + engagement_factor * 2.5)
            hotel_search_count = rng.poisson(0.2 + engagement_factor * 2.2)

            # --- Match ---
            match_types, match_probs = zip(*MATCH_TYPES)
            match_importance = rng.choice(match_types, p=match_probs)

            # Days until match: depends on match importance
            if match_importance in ("Semi-final", "Final"):
                days_until_match = int(rng.integers(30, 400))
            elif match_importance == "Quarter-final":
                days_until_match = int(rng.integers(20, 380))
            elif match_importance == "Round of 16":
                days_until_match = int(rng.integers(15, 360))
            elif match_importance == "Round of 32":
                days_until_match = int(rng.integers(10, 340))
            else:
                days_until_match = int(rng.integers(5, 365))

            # --- Trip cost ---
            # Based on distance, host country, match importance
            base_cost = distance_km * 0.22  # rough per-km cost
            if host_country == "USA":
                base_cost *= 1.15
            elif host_country == "Canada":
                base_cost *= 1.05
            # Add match premium
            match_mult = {
                "Group": 1.0, "Round of 32": 1.05, "Round of 16": 1.08,
                "Quarter-final": 1.12, "Semi-final": 1.18, "Final": 1.30,
            }
            base_cost *= match_mult[match_importance]
            base_cost += rng.normal(0, base_cost * 0.15)
            estimated_trip_cost = round(max(base_cost, 150), 2)

            # --- Visa ---
            visa_required = int(visa_needed(cname, host_country))
            # Host-country fans don't need visa
            if cname in ("United States", "Canada", "Mexico"):
                visa_required = 0

            # --- Campaign cost ---
            # Cost to target this fan: small variation around a base
            campaign_cost = round(5.0 + rng.uniform(0, 8) + (0.5 if visa_required else 0), 2)
            campaign_cost = min(campaign_cost, 15.0)

            # --- Potential revenue ---
            # Revenue if fan converts: based on trip cost, income, match importance
            rev_base = estimated_trip_cost * 0.35
            rev_match_mult = {
                "Group": 0.8, "Round of 32": 0.9, "Round of 16": 1.0,
                "Quarter-final": 1.15, "Semi-final": 1.25, "Final": 1.50,
            }
            rev_base *= rev_match_mult[match_importance]
            rev_base += rng.normal(0, rev_base * 0.2)
            potential_revenue = round(max(rev_base, 20), 2)

            # --- TARGET: will_travel ---
            # Realistic logit-based probability using real factors
            # Standardize continuous factors
            dist_z = (distance_km - 5000) / 4000  # rough z-score
            eng_z = (football_engagement - 50) / 25
            income_num = 3 if income_level == "High" else 2 if income_level == "Medium" else 1
            income_z = (income_num - 2) / 0.8
            match_num = {"Group": 1, "Round of 32": 2, "Round of 16": 3,
                         "Quarter-final": 4, "Semi-final": 5, "Final": 6}
            match_z = (match_num[match_importance] - 2.5) / 1.5

            logit = (
                -0.5                         # intercept (base rate)
                - 0.8 * dist_z               # farther → less likely
                + 0.9 * eng_z                # more engaged → more likely
                + 0.4 * income_z             # wealthier → more likely
                - 0.5 * visa_required        # visa → less likely
                + 0.6 * previous_attendance   # been before → more likely
                + 0.4 * favorite_team_qualified  # team qualified → more likely
                + 0.3 * match_z              # bigger match → more likely
                + rng.normal(0, 0.7)         # individual variation
            )
            travel_prob = 1.0 / (1.0 + math.exp(-logit))
            travel_prob = round(travel_prob, 4)

            # Binary target
            will_travel = int(rng.random() < travel_prob)

            # Expected value
            expected_value = round(travel_prob * potential_revenue - campaign_cost, 2)

            records.append({
                "fan_id": fan_id,
                "age": age,
                "country": cname,
                "country_region": region,
                "income_level": income_level,
                "gdp_per_capita_usd": cd["gdp"],
                "distance_to_host_city_km": distance_km,
                "nearest_host_city": host_city,
                "nearest_host_country": host_country,
                "favorite_team_qualified": favorite_team_qualified,
                "previous_worldcup_attendance": previous_attendance,
                "football_engagement_score": football_engagement,
                "social_media_engagement": social_media,
                "ticket_search_count": ticket_search_count,
                "flight_search_count": flight_search_count,
                "hotel_search_count": hotel_search_count,
                "estimated_trip_cost": estimated_trip_cost,
                "visa_required": visa_required,
                "days_until_match": days_until_match,
                "match_importance": match_importance,
                "campaign_cost_usd": campaign_cost,
                "potential_net_revenue_usd": potential_revenue,
                "expected_value_usd": expected_value,
                "travel_probability_synthetic": travel_prob,
                "will_travel": will_travel,
            })

    df = pd.DataFrame(records)
    return df


# ===========================================================================
# 6. MAIN
# ===========================================================================
def main():
    print("=" * 65)
    print("Generating Synthetic World Cup Fan Dataset")
    print("with Real-World Host Cities, Geography & Visa Data")
    print("=" * 65)
    print()

    df = generate_dataset(n_fans=50_000)

    print(f"Dataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Target distribution:")
    print(f"  will_travel = 1: {df['will_travel'].sum():,} ({df['will_travel'].mean():.1%})")
    print(f"  will_travel = 0: {(df['will_travel'] == 0).sum():,} ({(1 - df['will_travel'].mean()):.1%})")
    print()

    print("Country distribution (top 10):")
    print(df["country"].value_counts().head(10))
    print()

    print("Host city distribution (top 10):")
    print(df["nearest_host_city"].value_counts().head(10))
    print()

    print("Travel rate by region:")
    print(df.groupby("country_region")["will_travel"].mean().sort_values(ascending=False))
    print()

    print("Travel rate by visa requirement:")
    print(df.groupby("visa_required")["will_travel"].mean())
    print()

    # Save
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"✓ Dataset saved to: {OUTPUT_PATH}")

    # Also update the data dictionary
    data_dict = pd.DataFrame([
        ("fan_id", "Unique synthetic fan identifier"),
        ("age", "Fan age (18–80)"),
        ("country", "Fan's country of residence (real country)"),
        ("country_region", "Fan's geographic region"),
        ("income_level", "Income category (Low/Medium/High, derived from GDP/capita)"),
        ("gdp_per_capita_usd", "Real GDP per capita of fan's country"),
        ("distance_to_host_city_km", "Haversine distance from country centroid to nearest 2026 host city"),
        ("nearest_host_city", "Nearest official 2026 World Cup host city"),
        ("nearest_host_country", "Country of the nearest host city (USA/Canada/Mexico)"),
        ("favorite_team_qualified", "Whether fan's favorite team (national) qualified (1=Yes)"),
        ("previous_worldcup_attendance", "Whether fan previously attended a World Cup (1=Yes)"),
        ("football_engagement_score", "General football interest score (0–100)"),
        ("social_media_engagement", "World Cup social media engagement (0–100)"),
        ("ticket_search_count", "Recent ticket-related search count"),
        ("flight_search_count", "Recent flight-related search count"),
        ("hotel_search_count", "Recent hotel-related search count"),
        ("estimated_trip_cost", "Estimated total trip cost in USD"),
        ("visa_required", "Whether visa is required for host country entry (1=Yes, real policy)"),
        ("days_until_match", "Days until the target match"),
        ("match_importance", "Match stage (Group/Round of 32/Round of 16/Quarter-final/Semi-final/Final)"),
        ("campaign_cost_usd", "Estimated marketing campaign cost per person (USD)"),
        ("potential_net_revenue_usd", "Estimated potential net revenue if fan converts (USD)"),
        ("expected_value_usd", "travel_probability * potential_revenue - campaign_cost"),
        ("travel_probability_synthetic", "Hidden synthetic probability — NOT a model feature (data leakage)"),
        ("will_travel", "Target: 1 if fan travels/interested, 0 otherwise"),
    ], columns=["column", "description"])

    dict_path = DATA_DIR / "synthetic_worldcup_data_dictionary.csv"
    data_dict.to_csv(dict_path, index=False)
    print(f"✓ Data dictionary saved to: {dict_path}")


if __name__ == "__main__":
    main()
