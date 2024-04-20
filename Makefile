.ONESHELL:

quality:
	pre-commit run --all-files

remote_username=ubuntu
remote_ip=192.168.2.228
remote_data_path=
local_data_path=data/
update_dataset:
	rsync -ahP $(remote_username)@$(remote_ip):$(remote_data_path) $(local_data_path)

env_name=amar-wr
dev_env:
	conda create -n $(env_name) python=3.10 -y
	conda run -n $(env_name) --no-capture-output --live-stream pip install -e .
