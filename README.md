# SAFE-COLOR
Scripts for reproducing the results presented in the IEEE IV paper *SAFE-COLOR: Color Fidelity Benchmarks and Thresholds for Safety-Critical Object Detection* by Marvin Damschen and Ramana Reddy Avula and Mazen Mohamad. 

Preprint and dataset available: https://doi.org/10.5281/zenodo.14864429

If you use the scripts or dataset, please cite:

    @inproceedings{Damschen2025SafeColor,
    author = {Marvin Damschen and Ramana Reddy Avula and Mazen Mohamad},
    title = {{SAFE-COLOR}: Color Fidelity Benchmarks and Thresholds for Safety-Critical Object Detection},
    booktitle = {Proceedings of the 36th IEEE Intelligent Vehicles Symposium (IV 2025)},
    year = {2025},
    publisher = {IEEE}
    }


The scripts as described below will generate the dataset and evaluate YOLO using [Ultralytics](https://github.com/ultralytics/ultralytics).

## Prerequisites
- Download coco dataset to the datasets folder or create a symbolic link using, for example:

      ln -s /data/YOLO/datasets/coco/ ./datasets/coco

- Place the YOLO models in the models folder or use download_models.sh to download them.
- Install pip dependencies:

      pip install -r requirements.txt

## Usage
Run the ultralytics docker container by mounting the current directory:

    docker run -it --ipc=host --gpus all -v ./:/ultralytics/workspace -v ./datasets/:/datasets -w /ultralytics/workspace ultralytics/ultralytics /bin/bash -c "pip install --root-user-action ignore -r requirements.txt && exec /bin/bash"

If using a symbolic link for coco dataset, make sure to mount the coco dataset to the container, for example:

    docker run -it --ipc=host --gpus all -v ./:/ultralytics/workspace -v ./datasets/:/datasets -v /data/YOLO/datasets/coco/:/data/YOLO/datasets/coco -w /ultralytics/workspace ultralytics/ultralytics /bin/bash -c "pip install --root-user-action ignore -r requirements.txt && exec /bin/bash"

Run one of the run script in parent directory within the container:

    python coco_run_deltaE_gbch_range.py | tee results/coco_run_deltaE_gbch_range.log

Results will end up in the results folder, deltaE range needs currently to be changed within the script.
