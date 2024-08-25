# Setup
1. Use the Makefile target `make dev_env` to setup the environment. Add the conda env name inside the Makefile
2. You can also use the requirements.txt file to setup the environment using `pip install -r requirements.txt`

# Training
Entrypoint - `launch_task.py` and `conf/config.yaml`

For training, Change the task to `train` inside config.yaml and run `python launch_task.py`

For testing, Change the task to `test` inside config.yaml and provide the `wandb run's id`, set `resume=True` and provide the epoch to resume from

For exporting, Follow the same steps as testing but change the task to `export`

# Launching the app
Project uses triton server to deploy the model, it is exported as a onnx file which is not part of this repository but you can ask the owner for access.
* Run the bash script `launch_triton.sh` to start triton server like this
   ```
   bash launch_triton.sh
   ```
* Run the bash script `launch_app.sh` to start the FastAPI server like this
  ```
  bash launch_app.sh
  ```
