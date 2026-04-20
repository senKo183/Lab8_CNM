import h2o
import json
import random
import numpy as np
import os

from h2o.automl import H2OAutoML

# ==================================
# 1. Fix Random Seed
# ==================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# ==================================
# 2. Load Config
# ==================================

with open("config.json", "r") as f:
    config = json.load(f)

data_path = config["data_path"]
model_path = config["model_path"]
max_models = config["max_models"]
runtime = config["runtime"]

# ==================================
# 3. Start H2O
# ==================================

h2o.init()

# ==================================
# 4. Load Dataset
# ==================================

data = h2o.import_file(data_path)

target = "classification"
features = list(data.columns)
features.remove(target)

# ==================================
# 5. Train Test Split
# ==================================

train, test = data.split_frame(
    ratios=[0.8],
    seed=SEED
)

# ==================================
# 6. AutoML Training
# ==================================

aml = H2OAutoML(
    max_models=max_models,
    seed=SEED,
    max_runtime_secs=runtime
)

aml.train(
    x=features,
    y=target,
    training_frame=train
)

# ==================================
# 7. Best Model
# ==================================

leader = aml.leader
print("Best Model:", leader)

# ==================================
# 8. Evaluation
# ==================================

performance = leader.model_performance(test)

accuracy = 1 - performance.mean_per_class_error()

print("Accuracy:", accuracy)

# ==================================
# 9. Save Model
# ==================================

saved_model = h2o.save_model(
    model=leader,
    path=model_path,
    force=True
)

print("Model saved at:", saved_model)

# ==================================
# 10. Save Metrics
# ==================================

metrics = {
    "accuracy": float(accuracy)
}

with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print("Metrics saved")

# ==================================
# 11. Shutdown
# ==================================

h2o.shutdown(prompt=False)