# `myoadapt.tracking` — Experiment tracking

## Local tracker (no dependencies)

```python
from myoadapt.tracking import LocalTracker

tracker = LocalTracker('./experiments')
tracker.set_experiment('lite_dan_db2')
tracker.start_run(run_name='lite_dan_v1')

tracker.log_params({
 'model': 'lite_dan',
 'n_features': 308,
 'n_classes': 12,
 'lambda_schedule': 'gradual',
 'n_epochs': 100,
})

tracker.log_metric('accuracy', 0.72)
tracker.log_metric('macro_f1', 0.68)
tracker.log_artifact('./models/lite_dan.pkl')

tracker.end_run()

# Query
runs = tracker.list_runs('lite_dan_db2')
best = tracker.get_best_run('accuracy', mode='max')
```

## MLflow adapter (optional)

```python
from myoadapt.tracking import MLflowTracker, get_tracker

tracker = get_tracker(use_mlflow=True, tracking_uri='http://localhost:5000')
# Same API as LocalTracker, but logs to MLflow if available
```
