from ultralytics import YOLO
from ultralytics import settings
import os
import shutil

def prepare_coco_cfg(folder, transform, deltaE):
    source_file = os.path.join(folder, "coco.yaml")
    prefix = f"{transform}_deltaE{deltaE}_0_"
    new_file = os.path.join(folder, f"{prefix}coco.yaml")
    shutil.copy(source_file, new_file)

    with open(new_file, "r") as file:
        content = file.read()

    content = content.replace("coco", f"{prefix}coco")

    with open(new_file, "w") as file:
        file.write(content)

    # Copy or link val2017.txt
    shutil.copy(os.path.join(folder, "coco/val2017.txt"), os.path.join(folder, f"{prefix}coco/val2017.txt"))

    link_name = os.path.join(folder, f"{prefix}coco/images/")
    os.makedirs(os.path.dirname(link_name), exist_ok=True)

    # Symlink original images and annotations
    os.symlink( os.path.join(folder, "coco/images/train2017"), os.path.join(link_name, "train2017"))
    os.symlink( os.path.join(folder, "coco/images/test2017"), os.path.join(link_name, "test2017"))
    os.symlink( os.path.join(folder, "coco/labels"), os.path.join(folder, f"{prefix}coco/labels"))
    os.symlink( os.path.join(folder, "coco/annotations"), os.path.join(folder, f"{prefix}coco/annotations"))


deltaE_range = range(0, 54, 3)  # 0, 3, 6, ..., 51
script_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(script_dir, "datasets")
settings.update({"datasets_dir": datasets_dir})

models_dir = os.path.join(script_dir, "models")
model_filenames = os.listdir(models_dir)

predictions_dir = os.path.join(script_dir, "results/predictions")
metrics_dir = os.path.join(script_dir, "results/metrics")
os.makedirs(predictions_dir, exist_ok=True)
os.makedirs(metrics_dir, exist_ok=True)

# List of transforms for i>0
transforms = [
    "gamma-inc","gamma-dec",
    "brightness-inc","brightness-dec",
    "contrast-inc","contrast-dec",
    "hue","pull_mean",
    "saturation-inc","saturation-dec"
]

for model_filename_index, model_filename in enumerate(model_filenames):
    model_name = os.path.splitext(model_filename)[0]
    model = YOLO(os.path.join(models_dir, model_filename))

    # We will store the cached mAP for i==0 (no_transform) here.
    no_transform_map_for_i0 = None

    for i in deltaE_range:
        for transform in transforms:
            print(f"############ Model = {model_name}, DeltaE = {i}, Transform = {transform} ############")

            # Each transform writes into its own mAP result file
            result_txt_path = os.path.join(metrics_dir, f"{transform}_mAP_coco.txt")
            result_txt = open(result_txt_path, "a", encoding="utf-8")

            # Header row for the CSV in each transform's result_txt (only if first model)
            if model_filename_index == 0 and i == deltaE_range[0]:
                result_txt.write("model_name")
                for j in deltaE_range:
                    result_txt.write(f",deltaE={j}")
                result_txt.write('\n')

            # If this is the first time we write for this model in this file at i=0, put the model name
            if i == deltaE_range[0]:
                result_txt.write(model_name)

            # ----------------------------------------------------------------------------
            # 1) If i == 0 => Use "no_transform" only ONCE for dataset + val
            #    Then reuse the same mAP for all transforms.
            # ----------------------------------------------------------------------------
            if i == 0:
                # If we haven't run "no_transform" dataset + val yet, do it now:
                if no_transform_map_for_i0 is None:
                    # Only prepare new images and config if this is the first model
                    if model_filename_index == 0:
                        # Create images with no_transform
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} --transform no_transform"
                        )
                        # Prepare dataset config for no_transform
                        prepare_coco_cfg(datasets_dir, "no_transform", i)

                    # Validate
                    validation_results = model.val(
                        data=os.path.join(datasets_dir, f"no_transform_deltaE{i}_0_coco.yaml"),
                        project=os.path.join(predictions_dir, f"no_transform_deltaE{i}_0_coco"),
                        name=model_name,
                        save_json=True,
                        plots=True
                    )
                    # Cache the mAP
                    no_transform_map_for_i0 = validation_results.box.map

                # Write that cached mAP for every transform at i=0
                result_txt.write("," + str(no_transform_map_for_i0))

            # ----------------------------------------------------------------------------
            # 2) If i > 0 => Proceed with the normal transform logic
            # ----------------------------------------------------------------------------
            else:
                # Only prepare new images and config if this is the first model
                if model_filename_index == 0:
                    if transform == "gamma-inc":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform gamma --gamma_range 1.0 10.0"
                        )
                    elif transform == "gamma-dec":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform gamma --gamma_range 0.01 1.0"
                        )
                    elif transform == "brightness-inc":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform brightness --brightness_range 0.0 1.0"
                        )
                    elif transform == "brightness-dec":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform brightness --brightness_range -1.0 0.0"
                        )
                    elif transform == "contrast-inc":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform contrast --contrast_range 1.0 50.0"
                        )
                    elif transform == "contrast-dec":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform contrast --contrast_range 0.0 1.0"
                        )
                    elif transform == "saturation-inc":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform saturation --saturation_range 1.0 50.0"
                        )
                    elif transform == "saturation-dec":
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            "--transform saturation --saturation_range 0.0 1.0"
                        )
                    else:
                        os.system(
                            f"cd {datasets_dir}; "
                            f"python reduce_color_de_gbch_folder.py coco/images/val2017 --deltaE {i} "
                            f"--transform {transform}"
                        )

                    # Prepare the custom dataset config
                    prepare_coco_cfg(datasets_dir, transform, i)

                # Now validate using that new dataset
                validation_results = model.val(
                    data=os.path.join(datasets_dir, f"{transform}_deltaE{i}_0_coco.yaml"),
                    project=os.path.join(predictions_dir, f"{transform}_deltaE{i}_0_coco"),
                    name=model_name,
                    save_json=True,
                    plots=True
                )

                result_txt.write("," + str(validation_results.box.map))

            # If this was the last ΔE, end the line
            if i == deltaE_range[-1]:
                result_txt.write('\n')

            result_txt.flush()
            result_txt.close()

print("All done.")
