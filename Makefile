.ONESHELL:

quality:
	pre-commit run --all-files

project_name=jivi-ass

s3_uri=s3://ai-sagemaker-datasets/dojo/$(project_name)
local_data_path=data/raw/
update_dataset:
	aws s3 sync $(s3_uri) $(local_data_path)

remote_username=ubuntu
remote_ip=192.168.2.228
remote_data_path=
local_data_path=data/raw/
update_dataset_from_local:
	rsync -ahP $(local_data_path) $(remote_username)@$(remote_ip):$(remote_data_path)

env_name=$(project_name)
dev_env:
	conda create -n $(env_name) python=3.10 -y
	conda run -n $(env_name) --no-capture-output --live-stream pip install -e .

project_name=
sweep:
	python -m wandb sweep --project ${project_name} conf/sweep.yaml

entity=spyne-ai
sweep_id=
count=5
agent:
	python -m wandb agent --count ${count} ${entity}/${project_name}/${sweep_id}
