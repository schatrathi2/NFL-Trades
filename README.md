# NFL Draft Trades Analysis

This repository contains a Python script (`nfl_trades.py`) and a dataset (`trades.csv`) for analyzing and visualizing NFL draft pick trades.

## Features

- Load and preprocess trade data
- Normalize team abbreviations
- Calculate trade value based on pick number bins
- Build a directed multigraph of trades using NetworkX
- Visualize the trade network graph and save as `nfl_trade_network.png`
- Interactive command-line interface

## Data Format

The `trades.csv` file must include the following columns:
- `trade_id` (unique identifier)
- `gave` (abbreviation of the team giving the pick, e.g., 'ATL')
- `received` (abbreviation of the team receiving the pick)
- `season` (year of the draft, e.g., 2020)
- `pick_round` (round number of the pick, integer)
- `pick_number` (overall pick number, integer)

Team abbreviations for relocated franchises are normalized:
- `STL`, `LA` → `LAR`
- `OAK` → `LV`
- `SD` → `LAC`

### Interactive Options

- **1. Most related team**: Enter a team abbreviation to find which other team it has traded with the most.
- **2. Team trades count**: Enter a team and season to count trades participated.
- **3. Most connected team**: Identify the team with the highest total trade count.
- **4. Highest received value**: Show the trade(s) where a team received the highest pick value.
- **5. Exit**: Quit the program.

## Notes

- The script bins pick numbers into groups of size 16 to compute trade value. Lower pick numbers have higher values.
- Graph nodes represent teams; edges represent directed pick trades with edge weight proportional to pick value.