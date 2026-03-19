# Solver Parameters Configuration

## Overview

CFD and Solar solver parameters are loaded from a JSON configuration file to avoid hardcoding in code.

## Configuration File

### Location
`solver_parameters.json`

### Structure

```json
{
  "cfd": {
    "wind_speed": 2.0,           // Wind speed (m/s)
    "wind_direction": 45.0,     // Wind direction (degrees, 0=N, 90=E)
    "height": 2.0,              // Analysis height (m)
    "temperature": 28.0,        // Ambient temperature (°C)
    "humidity": 70.0,           // Relative humidity (%)
    "voxel_pitch": 1.0,         // Voxel size (m)
    "buffer_ratio": 1.5,        // Buffer ratio
    "alpha_t": 0.1,             // Temperature diffusion coefficient
    "alpha_rh": 0.1,           // Humidity diffusion coefficient
    "building_radius": 500.0   // Building influence radius (m)
  },
  "solar": {
    "time": "2025-10-04 14:00:00+08:00",  // Analysis time (ISO + timezone)
    "latitude": 1.379,                     // Latitude (°)
    "longitude": 103.893,                  // Longitude (°)
    "elevation": 14.0,                     // Elevation (m)
    "DNI": 800.0,                          // Direct normal irradiance (W/m²)
    "DHI": 180.0,                          // Diffuse horizontal irradiance (W/m²)
    "rays_per_receiver": 64,               // Rays per receiver
    "ground_radius": 25.0,                  // Ground radius (m)
    "shading_threshold": 0.1,              // Shading threshold
    "grid_resolution": 32                  // Grid resolution
  }
}
```

## Usage

### 1. In `intelligent_building_agent.py`

```python
from intelligent_building_agent import IntelligentBuildingAgent

# Specify config file path when creating the agent
agent = IntelligentBuildingAgent(
    api_key=OPENAI_API_KEY,
    config_file="/path/to/solver_parameters.json"
)

# Run analysis — parameters from JSON will be used
result = agent.analyze(
    query="Analyze wind flow around the building",
    stl_directory="/path/to/stl/files"
)
```

### 2. In `full_analysis_with_recording_en.py`

The config is already integrated; just run:

```bash
python full_analysis_with_recording_en.py
```

By default uses `solver_parameters.json` in the project root.

### 3. Parameter Priority

The system uses a three-level priority (lowest to highest):

1. **JSON config file** — Base defaults
2. **LLM analysis** — AI-recommended parameters from the query
3. **User-provided** — Highest priority

Example:

```python
# JSON config + user overrides
result = agent.analyze(
    query="Analyze wind flow",
    stl_directory="/path/to/stl",
    user_parameters={
        "cfd": {
            "wind_speed": 3.0,  # Override wind_speed from JSON
            # Other parameters still from JSON
        }
    }
)
```

## Editing Configuration

### Method 1: Edit JSON directly

```bash
nano solver_parameters.json
```

### Method 2: Multiple config files

Create separate configs for different scenarios:

```
solver_parameters_summer.json
solver_parameters_winter.json
solver_parameters_typhoon.json
```

Then specify in code:

```python
agent = IntelligentBuildingAgent(
    api_key=OPENAI_API_KEY,
    config_file="solver_parameters_summer.json"
)
```

## Parameter Reference

### CFD Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| wind_speed | float | 2.0 | Inflow wind speed (m/s) |
| wind_direction | float | 45.0 | Wind direction (degrees, 0=N, 90=E) |
| height | float | 2.0 | Pedestrian-level analysis height (m) |
| temperature | float | 28.0 | Ambient temperature (°C) |
| humidity | float | 70.0 | Relative humidity (%) |
| voxel_pitch | float | 1.0 | CFD voxel size (m) |
| buffer_ratio | float | 1.5 | Domain buffer ratio |
| alpha_t | float | 0.1 | Temperature diffusion coefficient |
| alpha_rh | float | 0.1 | Humidity diffusion coefficient |
| building_radius | float | 500.0 | Building influence radius (m) |

### Solar Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| time | string | "2025-10-04 14:00:00+08:00" | Analysis time (ISO + timezone) |
| latitude | float | 1.379 | Latitude (°N) |
| longitude | float | 103.893 | Longitude (°E) |
| elevation | float | 14.0 | Elevation (m) |
| DNI | float | 800.0 | Direct normal irradiance (W/m²) |
| DHI | float | 180.0 | Diffuse horizontal irradiance (W/m²) |
| rays_per_receiver | int | 64 | Ray-tracing density |
| ground_radius | float | 25.0 | Ground reflection radius (m) |
| shading_threshold | float | 0.1 | Shading threshold (0–1) |
| grid_resolution | int | 32 | Grid resolution |

## Example Scenarios

### Summer afternoon (Singapore)

```json
{
  "cfd": {
    "wind_speed": 1.5,
    "wind_direction": 135.0,
    "temperature": 32.0,
    "humidity": 75.0
  },
  "solar": {
    "time": "2025-06-21 14:00:00+08:00",
    "DNI": 900.0,
    "DHI": 200.0
  }
}
```

### Winter (Northern hemisphere)

```json
{
  "cfd": {
    "wind_speed": 3.0,
    "wind_direction": 0.0,
    "temperature": 10.0,
    "humidity": 50.0
  },
  "solar": {
    "time": "2025-12-21 12:00:00+08:00",
    "DNI": 600.0,
    "DHI": 150.0
  }
}
```

### Typhoon conditions

```json
{
  "cfd": {
    "wind_speed": 15.0,
    "wind_direction": 90.0,
    "temperature": 26.0,
    "humidity": 90.0
  }
}
```

## Debugging

If the config file fails to load, the system prints a warning:

```
⚠️  Configuration file not found: /path/to/config.json
⚠️  Invalid JSON in configuration file: ...
```

Execution continues using code defaults or LLM-suggested parameters.

## Related Files

- `solver_parameters.json` — Default config file
- `intelligent_building_agent.py` — Agent main file, config loading logic
- `full_analysis_with_recording_en.py` — Full analysis script, JSON config integrated
- `wrapper/cfd_solver.py` — CFD parameter wrapper
- `wrapper/solar_solver.py` — Solar parameter wrapper
