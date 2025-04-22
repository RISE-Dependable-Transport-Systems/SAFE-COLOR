from ultralytics import YOLO
from ultralytics import settings
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, "datasets")
settings.update({"datasets_dir": datasets_dir})

models_dir = os.path.join(script_dir, "models")
model_filenames = os.listdir(models_dir)

data_config_filepath = os.path.join(datasets_dir, "coco.yaml")
predictions_dir = os.path.join(script_dir, "results/predictions/clean_coco")

selected_models = []

for model_filename in model_filenames:
  model_name = os.path.splitext(model_filename)[0]
  if selected_models and model_name not in selected_models:
    continue

  model = YOLO(models_dir + model_filename)
  validation_results = model.val(data=data_config_filepath, project=predictions_dir, name=model_name, save_json=True, plots=True, device="cuda:0")