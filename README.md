# NGBoost Pressure Gradient Model

This package contains a trained probabilistic machine learning model for
predicting two-phase frictional pressure gradients in multiport tubes,
along with everything needed to load it and run predictions on new data —
no retraining required.

## Contents

| File | Description |
|---|---|
| `ngboost_pressure_gradient_model.pkl` | The trained NGBoost model (Normal distribution, RandomForest base learner), packaged with its feature list and inference settings. |
| `reviewer_predict_script.py` | Loads the model and computes predictions with 95% confidence intervals. |
| `README.md` | This file. |

## Setup

Install the exact package versions the model was trained and pickled with,
to avoid any pickle-compatibility issues:

```bash
pip install ngboost==0.5.11 scikit-learn==1.8.0 pandas numpy openpyxl
```

## Required Input Features

The model expects the following 7 dimensionless physical features,
**in this exact order** (the script enforces the order internally, so
your data just needs to contain these columns — any extra columns are
ignored):

1. `Frtp`
2. `Beta`
3. `Prtp`
4. `Refo`
5. `BoPh_Pf` — computed as `Bo * Ph_Pf`; please compute this upstream if
   your raw data has `Bo` and `Ph_Pf` as separate columns
6. `Wefo`
7. `Sugo`

## How to Run

Place `ngboost_pressure_gradient_model.pkl` and `reviewer_predict_script.py`
in the same folder, then either:

**Option A — run directly:**
```bash
python reviewer_predict_script.py
```
Edit the example `test_data` DataFrame near the bottom of the script with
your own values, or point it at an Excel file with
`pd.read_excel("your_file.xlsx", sheet_name="Sheet1")`.

**Option B — import the function into your own code:**
```python
from reviewer_predict_script import predict_with_uncertainty
import pandas as pd

your_data = pd.read_excel("your_test_data.xlsx")
results = predict_with_uncertainty(your_data)
print(results)
```

## Output

`predict_with_uncertainty()` returns a DataFrame with:

| Column | Meaning |
|---|---|
| `Predicted_Mean` | Point prediction of the pressure gradient, in original (untransformed) units |
| `Lower_Bound_95CI` | Lower bound of the 95% confidence interval |
| `Upper_Bound_95CI` | Upper bound of the 95% confidence interval |
| `Log_Standard_Deviation` | The model's underlying uncertainty estimate, in log-space (provided for reference/diagnostics) |

**Note:** the model was trained on the natural log of the target and the
confidence interval is computed in log-space before being converted back
to physical units — this is why the interval is asymmetric around the
mean. `Predicted_Mean` and the CI bounds are already in the correct,
original physical units; no further transformation is needed on your end.

## Questions

Please reach out to [Sadaf Mehdi @ smehdi@purdue.edu] with any issues running
the script or interpreting the output.

Development of the model-serialization workflow and prediction script was assisted by Gemini (Google)
