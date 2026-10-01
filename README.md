# YuvaIntern Week 4 Machine Learning

Predict Adelie, Chinstrap and Gentoo species from bill length, bill depth, flipper length and body mass. The training-only comparison selects logistic regression using five-fold macro F1. Held-out accuracy is 69/69 (100%) on one fixed split; this is not guaranteed future accuracy.

## Download and run
Download and extract **YuvaIntern_Week4_Complete_Package.zip** before running. It contains all required data, charts and tables. Loose code files on GitHub are for review; running them requires the data folder from the package.

```sh
python -m pip install -r requirements.txt
python analysis.py
```

Alternatively open analysis.ipynb in Jupyter from the extracted folder and run all cells. The script regenerates tables and charts, not the report. Python 3.12 was used. Seeds are fixed at 42.

## Contents
- Word report with methods, results, three evaluation charts and limitations
- Python script and equivalent notebook
- Source and modeling CSV files
- Exact train/test split membership, fold scores and test predictions
- Requirements and portal submission description

## Data and limits
Source: https://allisonhorst.github.io/palmerpenguins/ (CC0), collected by Kristen Gorman and Palmer Station LTER and distributed by Horst, Hill and Gorman. A 344-row snapshot from 30 September 2026 is included. Two measurement-missing rows are excluded. The 342 retained records split into 273 train and 69 test. Prior EDA used the same dataset; independence and external generalization are not established. See the report for uncertainty and limitations.
