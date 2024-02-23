from lightning.pytorch.loggers import WandbLogger


def initialize_wandb_logger(project: str, id: str, resume: bool):
    print("Initializing Weights & Biases logger", end="\n\n")

    return WandbLogger(project=project, id=id, resume=resume)
