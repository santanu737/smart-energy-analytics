# Data Notes: BDG2 Dataset

## 1. Data Dictionary

| File | Shape | Key Columns | Time Range | Missing Values (Total Cells) |
|---|---|---|---|---|
| `metadata.csv` | (1636, 32) | `building_id`, `site_id`, `primaryspaceusage`, `sqm`, `electricity` | N/A | 28,173 |
| `weather.csv` | (331,166, 10) | `timestamp`, `site_id`, `airTemperature`, `cloudCoverage` | 2016-01-01 to 2017-12-31 | 652,836 |
| `electricity.csv` | (17,544, 1,579) | `timestamp`, [building_ids...] | 2016-01-01 to 2017-12-31 | 1,312,095 |
| `electricity_cleaned.csv` | (17,544, 1,579) | `timestamp`, [building_ids...] | 2016-01-01 to 2017-12-31 | 2,471,853 |

**Note on Electricity Meter Marking:** In `metadata.csv`, the `electricity` column contains 'Yes' (1578) or `NaN` (58). All 1578 buildings marked 'Yes' have corresponding columns in `electricity.csv`. No extra buildings are present in the meter files.

## 2. Site Summary

| Site ID | Buildings | Elec Meters | Timezone | Top 3 Primary Space Usages |
|---|---|---|---|---|
| Bear | 92 | 92 | US/Pacific | Education (67), Entertainment (9), Public services (6) |
| Bobcat | 36 | 35 | US/Mountain | Education (18), Office (5), Entertainment (4) |
| Bull | 124 | 123 | US/Central | Education (63), Lodging/res (19), Entertainment (18) |
| Cockatoo | 124 | 117 | US/Eastern | Education (46), Lodging/res (29), Office (19) |
| Crow | 5 | 5 | US/Eastern | Education (5) |
| Eagle | 106 | 106 | US/Eastern | Office (40), Education (26), Entertainment (11) |
| Fox | 137 | 137 | US/Mountain | Education (63), Office (24), Entertainment (21) |
| Gator | 74 | 74 | US/Eastern | Public services (29), Entertainment (26), Other (9) |
| Hog | 163 | 152 | US/Central | Office (77), Education (26), Lodging/res (11) |
| Lamb | 147 | 146 | Europe/London | Education (83), Entertainment (29), Office (18) |
| Moose | 15 | 13 | US/Eastern | Education (15) |
| Mouse | 7 | 7 | Europe/London | Healthcare (5), Lodging/res (1), Tech/science (1) |
| Panther | 136 | 105 | US/Eastern | Lodging/res (44), Education (35), Office (28) |
| Peacock | 47 | 45 | US/Eastern | Education (17), Office (12), Lodging/res (12) |
| Rat | 305 | 305 | US/Eastern | Education (111), Public services (93), Entertainment (47) |
| Robin | 52 | 52 | Europe/London | Education (22), Office (17), Lodging/res (10) |
| Shrew | 9 | 9 | Europe/London | Office (9) |
| Swan | 21 | 19 | US/Pacific | (All NaN) |
| Wolf | 36 | 36 | Europe/Dublin | Education (20), Office (9), Retail (3) |

## 3. Timezone and Frequency
- **Timestamps:** Confirmed to be sorted and unique in `electricity.csv`.
- **Frequency:** Exactly 1 hour apart (timedelta 01:00:00) for all 17,544 steps.
- **Timezone:** `metadata.csv` provides local timezones (e.g., US/Eastern, Europe/London). However, CSV timestamps are naive. **Crucial:** Do not perform timezone conversions unless evidence of misalignment is found during Phase 1/2.

## 4. Weather Data
- **Sites:** 19 sites, matching metadata.
- **Coverage:** Ranges from 16,860 to 17,544 rows per site. Some sites have missing hours in the weather file.

## 5. Things to verify in Phase 1/2
- **Weather Coverage:** Identify specific gaps per site to determine if interpolation is sufficient.
- **Raw vs Cleaned Difference:** Perform cell-level comparison to see if `electricity_cleaned.csv` mostly removes outliers/spikes or if it also drops valid low-consumption periods.
- **Zero Readings:** Investigate if zero values are concentrated in specific buildings or times (potential sensor failure vs. real zero usage).
