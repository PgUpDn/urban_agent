# Configuration Examples

This directory contains preset configuration files for different scenarios.

## Available Configs

### 1. `summer_afternoon.json` — Summer afternoon

For Singapore summer afternoon high-temperature conditions.

**Features:**
- Temperature: 33°C
- Humidity: 75%
- Southeast wind (135°), 1.5 m/s
- High solar radiation (DNI: 850 W/m²)
- Time: 21 June, 3:00 PM

**Use cases:**
- Extreme heat analysis
- Afternoon heat island studies
- Outdoor comfort assessment

### 2. `winter_morning.json` — Winter morning

For Singapore winter cool morning.

**Features:**
- Temperature: 24°C
- Humidity: 65%
- North wind (0°), 2.5 m/s
- Moderate solar radiation (DNI: 650 W/m²)
- Time: 21 December, 10:00 AM

**Use cases:**
- Coolest conditions of the year
- Natural ventilation potential
- Solar angle studies

### 3. `typhoon_conditions.json` — Typhoon conditions

For typhoon or storm weather.

**Features:**
- Temperature: 26°C
- Humidity: 90%
- East wind (90°), 12.0 m/s
- Low solar radiation (DNI: 300 W/m²) — overcast
- Time: 15 September, noon

**Use cases:**
- Extreme wind load analysis
- Pedestrian safety in strong wind
- Wind resistance assessment

## Usage

### Method 1: Use an example config directly

```python
from intelligent_building_agent import IntelligentBuildingAgent
from config import OPENAI_API_KEY

# Use summer config
agent = IntelligentBuildingAgent(
    api_key=OPENAI_API_KEY,
    config_file="config_examples/summer_afternoon.json"
)

result = agent.analyze(
    query="Analyze thermal comfort",
    stl_directory="/path/to/stl"
)
```

### Method 2: Copy and modify

```bash
# Copy example as main config
cp config_examples/summer_afternoon.json solver_parameters.json

# Edit parameters
nano solver_parameters.json
```

### Method 3: Batch analysis for multiple scenarios

```python
scenarios = [
    ("summer", "config_examples/summer_afternoon.json"),
    ("winter", "config_examples/winter_morning.json"),
    ("typhoon", "config_examples/typhoon_conditions.json")
]

for name, config in scenarios:
    agent = IntelligentBuildingAgent(
        api_key=OPENAI_API_KEY,
        config_file=config
    )

    result = agent.analyze(
        query=f"Environmental analysis — {name} conditions",
        stl_directory="/path/to/stl",
        output_directory=f"results/{name}"
    )
```

## Parameter Comparison

| Parameter | Summer afternoon | Winter morning | Typhoon |
|-----------|------------------|----------------|---------|
| Temperature (°C) | 33.0 | 24.0 | 26.0 |
| Humidity (%) | 75 | 65 | 90 |
| Wind speed (m/s) | 1.5 | 2.5 | 12.0 |
| Wind dir (°) | 135 (SE) | 0 (N) | 90 (E) |
| DNI (W/m²) | 850 | 650 | 300 |
| DHI (W/m²) | 200 | 160 | 150 |

## Custom Configuration

Create your own config from these examples:

```bash
# 1. Copy example
cp config_examples/summer_afternoon.json config_examples/my_custom.json

# 2. Edit parameters
nano config_examples/my_custom.json

# 3. Use
python your_analysis_script.py --config config_examples/my_custom.json
```

## Other Regions

For other locations, adjust the solar block:

### Beijing (China)

```json
{
  "solar": {
    "latitude": 39.9,
    "longitude": 116.4,
    "elevation": 43.0
  }
}
```

### New York (USA)

```json
{
  "solar": {
    "latitude": 40.7,
    "longitude": -74.0,
    "elevation": 10.0
  }
}
```

### London (UK)

```json
{
  "solar": {
    "latitude": 51.5,
    "longitude": -0.1,
    "elevation": 11.0
  }
}
```

## Tips

1. **Timezone:** Include correct offset in the time string, e.g. `+08:00` (Singapore/Beijing).
2. **Wind direction:** 0° = North, 90° = East, 180° = South, 270° = West.
3. **Radiation:** DNI (direct) + DHI (diffuse) ≈ GHI (global horizontal).
4. **Validation:** Use `verify_config.py` to validate modified configs.

## References

- [Solar parameters documentation](../solver_parameters_README.md)
- [Main configuration guide](../USAGE_JSON_CONFIG.md)
- [Intelligent Building Agent](../intelligent_building_agent.py)

---

**Last updated:** 2025-11-03
