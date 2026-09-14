"""
Deep ensemble model for two-phase frictional pressure gradient (dp/dz) prediction
in multiport tubes.

Five probabilistic neural networks are trained together, each predicting a mean
and a variance. Their disagreement gives you epistemic uncertainty (how much the
model itself is unsure), and their individual variances give you aleatoric
uncertainty (noise inherent to the data). Combined and calibrated, they form the
95% confidence interval returned by `predict`.

Use it as a library:

    from DEModel import load_ensemble, predict

    ensemble = load_ensemble("FINAL_DeepEnsemble_Deployed.pt")
    result = predict(ensemble, X)   # X: (n_samples, 7) in the feature order below

Or run it directly on an Excel file:

    python DEModel.py --model FINAL_DeepEnsemble_Deployed.pt --input my_data.xlsx

Add --target-column if your file also has true values, to get MAPE/R2/PICP@95:

    python DEModel.py --model FINAL_DeepEnsemble_Deployed.pt --input my_data.xlsx --target-column dpdzfr_Exp
"""

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _init_layer(layer):
    # truncated-normal init, same as what the model was trained with
    input_dim = layer.weight.shape[1]
    std = 1.0 / (2.0 * np.sqrt(input_dim))
    nn.init.trunc_normal_(layer.weight, std=std, a=-2 * std, b=2 * std)
    nn.init.zeros_(layer.bias)


def _weight_decay_groups(layer, decay):
    decay_params, no_decay_params = [], []
    for name, p in layer.named_parameters():
        (decay_params if "weight" in name else no_decay_params).append(p)
    return [
        {"params": decay_params, "weight_decay": decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class MeanStdevFilter:
    """Running mean/stdev normalizer. Fit once on training data, reused at inference."""

    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.stdev = np.ones(shape)

    def update(self, x):
        x = np.atleast_1d(np.array(x, dtype=float))
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.S += delta * (x - self.mean)
        if self.n > 1:
            self.stdev = np.sqrt(np.maximum(self.S / (self.n - 1), 1e-8))

    def fit(self, X):
        for row in X:
            self.update(row)
        return self

    def filter(self, x):
        return (x - self.mean) / (self.stdev + 1e-8)

    def invert_torch(self, x_norm):
        mean = torch.tensor(self.mean, dtype=torch.float32, device=x_norm.device)
        stdev = torch.tensor(self.stdev, dtype=torch.float32, device=x_norm.device)
        return x_norm * stdev + mean


class GaussianNLLLoss(nn.Module):
    """Negative log-likelihood for a Gaussian with predicted mean and log-variance.
    Only used during training, but the ensemble members hold a reference to it."""

    def forward(self, preds, targets, weights=None, logvar_loss=True):
        mu, logvar = preds.chunk(2, dim=1)
        targets = targets.reshape(-1, 1)
        if not logvar_loss:
            return F.mse_loss(mu, targets)
        var = logvar.exp()
        sq_err = (targets - mu).pow(2)
        per_sample = 0.5 * (logvar + sq_err / (var + 1e-8))
        if weights is None:
            return per_sample.mean()
        return (per_sample * weights.reshape(-1, 1)).sum() / weights.sum().clamp(min=1e-8)


class ProbabilisticNN(nn.Module):
    """A single ensemble member. Outputs a mean and a bounded log-variance."""

    def __init__(self, input_dim, output_dim, num_layers, hidden, l2, seed,
                 activation="elu", dropout=0.0):
        super().__init__()
        torch.manual_seed(seed)

        act_map = {"relu": nn.ReLU(), "silu": nn.SiLU(), "elu": nn.ELU()}
        self.activation = act_map.get(activation, nn.ELU())
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        self.fc1 = nn.Linear(input_dim, hidden)
        _init_layer(self.fc1)
        self.layers = nn.ModuleList([self.fc1])
        for _ in range(num_layers - 1):
            layer = nn.Linear(hidden, hidden)
            _init_layer(layer)
            self.layers.append(layer)

        self.delta = nn.Linear(hidden, output_dim)
        _init_layer(self.delta)
        self.logvar = nn.Linear(hidden, output_dim)
        _init_layer(self.logvar)

        self.layers_all = list(self.layers) + [self.delta, self.logvar]
        self.train_loss = GaussianNLLLoss()
        self.val_loss = GaussianNLLLoss()
        self.max_logvar, self.min_logvar = None, None

        # per-layer weight decay, heavier near the output
        decays = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001, 0.0001]
        while len(decays) < len(self.layers_all):
            decays.insert(-2, decays[-3])
        self.decays = np.array(decays[: len(self.layers_all)]) * l2

        params = []
        for layer, decay in zip(self.layers_all, self.decays):
            params.extend(_weight_decay_groups(layer, decay))
        self.weights = params
        self.to(DEVICE)

    def update_logvar_limits(self, max_lv, min_lv):
        self.max_logvar, self.min_logvar = max_lv, min_lv

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        logvar = self.max_logvar - F.softplus(self.max_logvar - self.logvar(x))
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return torch.cat((self.delta(x), logvar), dim=1)


class Model(nn.Module):
    """Wraps a ProbabilisticNN so the ensemble can hold a uniform list of these."""

    def __init__(self, input_dim, output_dim, num_layers, hidden, l2, seed,
                 activation="elu", dropout=0.0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = ProbabilisticNN(input_dim, output_dim, num_layers, hidden, l2,
                                      seed, activation, dropout)

    def forward(self, x):
        return self.model(x)


def load_ensemble(checkpoint_path, device=DEVICE):
    """Rebuild a trained ensemble from a .pt checkpoint, ready to call `predict` on."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    features = ckpt["features"]

    in_filt = MeanStdevFilter(len(features))
    in_filt.mean, in_filt.stdev = ckpt["in_filt_mean"], ckpt["in_filt_stdev"]

    out_filt = MeanStdevFilter(1)
    out_filt.mean, out_filt.stdev = ckpt["out_filt_mean"], ckpt["out_filt_stdev"]

    max_lv = ckpt["max_logvar"].to(device)
    min_lv = ckpt["min_logvar"].to(device)

    models = {}
    for i in range(cfg["num_models"]):
        m = Model(len(features), 1, cfg["num_layers"], cfg["num_nodes"], cfg["l2_multiplier"],
                  seed=i, activation=cfg["activation"], dropout=cfg["dropout"])
        m.load_state_dict(ckpt["ensemble_state_dicts"][i])
        m.model.update_logvar_limits(max_lv, min_lv)
        m.model.eval()
        models[i] = m

    return {
        "models": models,
        "in_filt": in_filt,
        "out_filt": out_filt,
        "T_cal": ckpt["T_cal"],
        "features": features,
        "device": device,
    }


def predict(ensemble, X_raw):
    """
    Predict dp/dz with calibrated uncertainty.

    X_raw: array-like of shape (n_samples, n_features), physical units,
           columns in the order given by ensemble["features"].

    Returns a dict of 1D numpy arrays: prediction, aleatoric_std, epistemic_std,
    total_std, ci_lower_95, ci_upper_95 — all in the same physical units as
    the training target.
    """
    device = ensemble["device"]
    in_filt, out_filt = ensemble["in_filt"], ensemble["out_filt"]
    T_cal = ensemble["T_cal"]

    X_log = np.log(np.asarray(X_raw, dtype=np.float32))
    X_norm = in_filt.filter(X_log).astype(np.float32)
    X_t = torch.tensor(X_norm, dtype=torch.float32, device=device)

    mus, logvars = [], []
    with torch.no_grad():
        for m in ensemble["models"].values():
            mu, lv = m.forward(X_t).chunk(2, dim=1)
            mus.append(mu)
            logvars.append(lv)

    mu_all = torch.stack(mus, 0)
    var_all = torch.stack(logvars, 0).exp()
    mu_norm = mu_all.mean(0)
    aleatoric_norm = var_all.mean(0)
    epistemic_norm = ((mu_all - mu_norm.unsqueeze(0)) ** 2).mean(0)

    s = float(out_filt.stdev[0])
    mu_log = np.atleast_1d(out_filt.invert_torch(mu_norm).squeeze().detach().cpu().numpy())
    aleatoric_std_log = np.atleast_1d(np.sqrt(aleatoric_norm.squeeze().detach().cpu().numpy()) * s)
    epistemic_std_log = np.atleast_1d(np.sqrt(epistemic_norm.squeeze().detach().cpu().numpy()) * s)
    total_std_log_cal = T_cal * np.sqrt(aleatoric_std_log ** 2 + epistemic_std_log ** 2)

    mu_phys = np.exp(mu_log)

    return {
        "prediction": mu_phys,
        "aleatoric_std": mu_phys * aleatoric_std_log,
        "epistemic_std": mu_phys * epistemic_std_log,
        "total_std": mu_phys * total_std_log_cal,
        "ci_lower_95": np.exp(mu_log - 1.96 * total_std_log_cal),
        "ci_upper_95": np.exp(mu_log + 1.96 * total_std_log_cal),
    }


def _run_cli():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="path to the .pt checkpoint")
    parser.add_argument("--input", required=True, help="Excel file with the feature columns")
    parser.add_argument("--sheet", default=0, help="sheet name or index in the input file")
    parser.add_argument("--output", default="predictions.xlsx", help="where to write results")
    parser.add_argument("--target-column", default=None,
                         help="optional column of true values, for reporting accuracy")
    args = parser.parse_args()

    ensemble = load_ensemble(args.model)
    features = ensemble["features"]

    df = pd.read_excel(args.input, sheet_name=args.sheet)
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"input file is missing required columns: {missing}")

    results = predict(ensemble, df[features].values)
    for name, values in results.items():
        df[name] = values

    if args.target_column:
        if args.target_column not in df.columns:
            raise ValueError(f"--target-column '{args.target_column}' not found in the input file")
        y_true = df[args.target_column].values.astype(float)
        y_pred = results["prediction"]

        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        picp95 = np.mean((y_true >= results["ci_lower_95"]) & (y_true <= results["ci_upper_95"]))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot

        print(f"MAPE:    {mape:.2f}%")
        print(f"R2:      {r2:.4f}")
        print(f"PICP@95: {picp95:.3f}")

    df.to_excel(args.output, index=False)
    print(f"saved predictions to {args.output}")


if __name__ == "__main__":
    _run_cli()
