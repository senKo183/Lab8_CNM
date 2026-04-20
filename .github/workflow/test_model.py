import json

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

assert accuracy > 0.8

print("Model test passed")