from typing import Optional

from .lit_module import ClassificationLitModule


def initialize_classification_lit_module(resume_ckpt_fpath: Optional[str] = None, **kwargs):
    if resume_ckpt_fpath is not None:
        print(f"Resuming from checkpoint: {resume_ckpt_fpath}")
        return ClassificationLitModule.load_from_checkpoint(resume_ckpt_fpath)
    else:
        return ClassificationLitModule(**kwargs)
