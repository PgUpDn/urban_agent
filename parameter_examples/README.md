# Parameter File Examples

This directory contains example JSON files produced by the parameter-tracking feature.

## File Descriptions

### 1. `llm_suggested_parameters_example.json`

Parameters suggested by the LLM after analyzing the user query.

**Scenario:** User query: "Analyze wind flow around the building with strong typhoon conditions"

**LLM analysis:**
- Identifies "typhoon conditions"
- Suggests higher wind speed (15.0 m/s)
- Suggests east wind (90°) — typical for Southeast Asian typhoons
- Sets temperature and humidity to typical typhoon values

**Structure:**
```json
{
  "source": "LLM Analysis",
  "timestamp": "2025-11-03T14:30:15.123456",
  "query": "Analyze wind flow around the building with strong typhoon conditions",
  "reasoning": "Based on the query mentioning 'strong typhoon conditions'...",
  "required_solvers": ["geometry", "cfd"],
  "parameters": {
    "cfd": {
      "wind_speed": 15.0,
      "wind_direction": 90.0,
      "height": 2.0,
      "temperature": 26.0,
      "humidity": 90.0
    }
  }
}
```

**Key fields:**
- `query`: Original user query
- `reasoning`: LLM reasoning (why these parameters)
- `required_solvers`: Solvers the LLM decided are needed
- `parameters`: Suggested parameter values

---

### 2. `final_cfd_parameters_example.json`

Final CFD parameters actually used (after merging).

**Sources:**
1. **JSON config** (`solver_parameters.json`):
   - `voxel_pitch: 1.0`
   - `buffer_ratio: 1.5`
   - `alpha_t: 0.1`
   - `alpha_rh: 0.1`
   - `building_radius: 500.0`

2. **LLM suggestions** (override JSON):
   - `wind_speed: 15.0` → later overridden by user to 12.0
   - `wind_direction: 90.0` → overridden by user to 180.0
   - `height: 2.0`, `temperature: 26.0`, `humidity: 90.0`

3. **User input** (highest priority):
   - `wind_speed: 12.0`
   - `wind_direction: 180.0`

**Structure:**
```json
{
  "source": "Final Merged Parameters",
  "timestamp": "2025-11-03T14:30:20.654321",
  "priority_order": [
    "1. JSON config file (base)",
    "2. LLM suggestions (override)",
    "3. User parameters (final override)"
  ],
  "cfd": {
    "wind_speed": 12.0,        // user override of LLM 15.0
    "wind_direction": 180.0,   // user override of LLM 90.0
    "height": 2.0,             // from LLM
    "temperature": 26.0,       // from LLM
    "humidity": 90.0,          // from LLM
    "voxel_pitch": 1.0,        // from JSON
    "buffer_ratio": 1.5,       // from JSON
    "alpha_t": 0.1,            // from JSON
    "alpha_rh": 0.1,          // from JSON
    "building_radius": 500.0   // from JSON
  }
}
```

**Notes:**
- Contains the full parameter set
- Priority is documented
- Timestamp records when it was generated

---

### 3. `final_solar_parameters_example.json`

Final solar parameters actually used.

**Scenario:** Summer solstice afternoon solar analysis.

**Sources:**
- Most from JSON config
- LLM may adjust time to solstice
- DNI adjusted for summer high radiation

**Structure:**
```json
{
  "source": "Final Merged Parameters",
  "timestamp": "2025-11-03T14:30:25.789012",
  "priority_order": [
    "1. JSON config file (base)",
    "2. LLM suggestions (override)",
    "3. User parameters (final override)"
  ],
  "solar": {
    "time": "2025-06-21 14:00:00+08:00",  // summer solstice 2 PM
    "latitude": 1.379,                     // Singapore
    "longitude": 103.893,
    "elevation": 14.0,
    "DNI": 850.0,                          // high direct (summer)
    "DHI": 200.0,
    "rays_per_receiver": 64,
    "ground_radius": 25.0,
    "shading_threshold": 0.1
  }
}
```

---

## How to Use These Examples

### 1. View example files

```bash
# LLM suggestions
cat parameter_examples/llm_suggested_parameters_example.json | jq

# Final CFD parameters
cat parameter_examples/final_cfd_parameters_example.json | jq

# Final solar parameters
cat parameter_examples/final_solar_parameters_example.json | jq
```

### 2. Compare LLM vs final parameters

```bash
# LLM CFD params
jq '.parameters.cfd' parameter_examples/llm_suggested_parameters_example.json

# Final CFD params
jq '.cfd' parameter_examples/final_cfd_parameters_example.json

# Diff
diff <(jq '.parameters.cfd' parameter_examples/llm_suggested_parameters_example.json) \
     <(jq '.cfd' parameter_examples/final_cfd_parameters_example.json)
```

### 3. Extract LLM reasoning

```bash
jq -r '.reasoning' parameter_examples/llm_suggested_parameters_example.json
```

### 4. Inspect parameter changes

```bash
# LLM wind speed
jq '.parameters.cfd.wind_speed' parameter_examples/llm_suggested_parameters_example.json
# → 15.0

# Final wind speed
jq '.cfd.wind_speed' parameter_examples/final_cfd_parameters_example.json
# → 12.0 (user override)
```

---

## Parameter Flow Example (typhoon)

**1. User query**
```
"Analyze wind flow around the building with strong typhoon conditions"
```

**2. LLM analysis** → `llm_suggested_parameters_example.json`
```
Keywords: "strong typhoon conditions"
→ wind_speed: 15.0 m/s
→ wind_direction: 90.0° (east)
→ temperature: 26.0°C, humidity: 90%
```

**3. Load JSON config**
```
voxel_pitch: 1.0, buffer_ratio: 1.5, alpha_t: 0.1, ...
```

**4. User overrides**
```python
user_parameters = {
    "cfd": {
        "wind_speed": 12.0,
        "wind_direction": 180.0
    }
}
```

**5. Final parameters** → `final_cfd_parameters_example.json`
```
wind_speed: 12.0        ← user
wind_direction: 180.0   ← user
height: 2.0             ← LLM
temperature: 26.0       ← LLM
humidity: 90.0          ← LLM
voxel_pitch: 1.0        ← JSON
buffer_ratio: 1.5       ← JSON
...
```

---

## Example Use Cases

### Review LLM suggestions

```python
import json

with open('parameter_examples/llm_suggested_parameters_example.json') as f:
    llm = json.load(f)

print(f"Query: {llm['query']}")
print(f"\nLLM Reasoning:\n{llm['reasoning']}")
print(f"\nSuggested wind speed: {llm['parameters']['cfd']['wind_speed']} m/s")

if llm['parameters']['cfd']['wind_speed'] > 20.0:
    print("⚠️  Warning: Wind speed seems too high!")
```

### Validate final parameters

```python
with open('parameter_examples/final_cfd_parameters_example.json') as f:
    final = json.load(f)

cfd = final['cfd']
print(f"Wind speed: {cfd['wind_speed']} m/s")
print(f"Wind direction: {cfd['wind_direction']}°")
assert 0 <= cfd['wind_speed'] <= 30, "Wind speed out of range"
assert 0 <= cfd['wind_direction'] < 360, "Wind direction out of range"
print("✅ All parameters validated")
```

### Save a good config as template

```python
import shutil

shutil.copy(
    'parameter_examples/final_cfd_parameters_example.json',
    'config_examples/typhoon_scenario_validated.json'
)
print("✅ Saved as new configuration template")
```

---

## Tips

1. **LLM files** show how the LLM interpreted your query.
2. **Final parameter files** show what was actually used at runtime.
3. Comparing them shows where each value came from.
4. Use these examples to understand how parameter tracking works.

---

## Related Docs

- [Parameter tracking details](../PARAMETER_TRACKING.md)
- [JSON config usage](../USAGE_JSON_CONFIG.md)
- [Parameter reference](../solver_parameters_README.md)

---

**Last updated:** 2025-11-03
