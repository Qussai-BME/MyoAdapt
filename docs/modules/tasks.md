# `myoadapt.tasks` — Higher-level tasks

## Continuous EMG decoding

Goes beyond discrete gesture classification to continuous outputs:

- Joint angle regression
- Force level prediction
- Continuous kinematics

Continuous decoding supports applications such as surgical-robotics
control and advanced prosthetics, where discrete gesture labels are
insufficient and a fluid, real-valued output is required.

### LSTM regressor

```python
from myoadapt.tasks import LSTMRegressor

model = LSTMRegressor(
    n_channels=12,
    n_outputs=5,        # 5-DOF hand kinematics
    hidden_dim=64,
    n_layers=2,
    n_epochs=50,
)
# X: (n_samples, seq_len, n_channels)
# y: (n_samples, n_outputs) — continuous kinematics
model.fit(X, y_kinematics)
preds = model.predict(X_test)
metrics = model.evaluate(X_test, y_test)
print(metrics)  # {'mse': ..., 'rmse': ..., 'r2': ..., 'mae': ...}
```

### Transformer regressor

```python
from myoadapt.tasks import TransformerRegressor

model = TransformerRegressor(
    n_channels=12,
    n_outputs=5,
    hidden_dim=64,   # d_model
    n_heads=4,
    n_layers=2,
)
model.fit(X, y_kinematics)
```

### NinaPro DB2 Exercise D (continuous force)

NinaPro DB2's Exercise D contains continuous force data, ideal for
training these regressors.

## Extensibility

The `ContinuousDecoder` base class is registered with the model
registry, so custom regression heads can be added by subclassing it and
registering with `@register_model("my_regressor")`. The same model
factory pattern used by the LOSO/LODO evaluators applies.
