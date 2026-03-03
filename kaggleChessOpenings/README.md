# Chess Openings Analysis – Data Mining Project

Analysis of chess opening statistics using association rules and clustering techniques to discover strategic patterns.

## Objective

Identify frequent move combinations in the first three moves of chess openings and group openings according to their statistical outcomes (win/draw rates for white/black).

## Dataset

- Source: [Kaggle – All Chess Openings](https://www.kaggle.com/datasets/alexandrelemercier/all-chess-openings)
- File: `openings.csv`
- ~2,500 openings with ECO code, move sequences, game counts, win/draw percentages, etc.

## Techniques Used

- **Association Rules** (Apriori + mlxtend) → frequent initial move patterns
- **K-Means Clustering** → grouping openings by outcome similarity (Player Win %, Opponent Win %, Draw %)
- Visualizations: bar plots, histograms, 2D/3D scatter plots, elbow method

## Main Findings

- Strong co-occurrences in early moves (e.g. 1.e4 → ... Nf6, 1.d4 → ... Nf6)
- Three natural clusters:
  - White-favoring aggressive lines
  - Balanced / drawish positions
  - Black-solid / counter-attacking defenses

## Requirements

```bash
pip install pandas numpy seaborn matplotlib scikit-learn mlxtend kagglehub
```

## How to run

Download the dataset from Kaggle or use kagglehub
Place openings.csv in the working directory (or adjust path)
Run the script top to bottom (preferably in Jupyter/Colab)

## Author
Cris N.
March 2025