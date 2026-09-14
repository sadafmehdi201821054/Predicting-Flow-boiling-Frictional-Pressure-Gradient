# DE\_MMC

A deep ensemble model for predicting two-phase frictional pressure gradient (dp/dz) in multiport tubes, with calibrated 95% confidence intervals.

Trained on flow boiling data across ten fluids: CO2, R134a, R32, R410A, R448A, R450A, R452B, R454B, R455A, and water. Five networks are trained together — their spread gives you epistemic uncertainty (model doubt), and each one's own predicted variance gives you aleatoric uncertainty (data noise). Both get combined and temperature-calibrated into the confidence interval you see in the output.

## What's in here

* `DEModel.py` — the network architecture, `load\_ensemble()`/`predict()`, and a command-line interface
* `FINAL\_DeepEnsemble\_Deployed.pt` — the trained weights
* `requirements.txt`

## Setup

```bash
pip install -r requirements.txt
```

## Running it on your own data

Your Excel file needs a column for each of these, named exactly (order doesn't matter, they're matched by name):

```
Frtp, Wefo, Refo, Sugo, Prtp, Beta, BoPh\_Pf
```

Then:

```bash
python DEModel.py --model FINAL\_DeepEnsemble\_Deployed.pt --input your\_data.xlsx --output predictions.xlsx
```

Got true dp/dz values in there too? Pass the column name and you'll get MAPE, R², and PICP@95 printed to the console:

```bash
python DEModel.py --model FINAL\_DeepEnsemble\_Deployed.pt --input your\_data.xlsx --target-column dpdzfr\_Exp
```

## Output columns

|Column|Meaning|
|-|-|
|`prediction`|predicted dp/dz (Pa/m)|
|`aleatoric\_std`|uncertainty from noise in the data itself|
|`epistemic\_std`|uncertainty from the model — goes up when you're outside what it was trained on|
|`total\_std`|combined, calibrated uncertainty|
|`ci\_lower\_95` / `ci\_upper\_95`|95% confidence interval|

## Using it directly in Python

```python
from DEModel import load\_ensemble, predict

ensemble = load\_ensemble("FINAL\_DeepEnsemble\_Deployed.pt")
result = predict(ensemble, X)  # X: array of shape (n\_samples, 7), same feature order as above

result\["prediction"]     # point estimate
result\["ci\_lower\_95"]    # lower bound
result\["ci\_upper\_95"]    # upper bound
```

## Background

Deep ensembles: Lakshminarayanan et al. (2017). Calibration via temperature scaling: Kuleshov et al. (2018).

