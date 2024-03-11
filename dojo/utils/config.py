import os


def get_resume_ckpt_fpath(cfg, exp_dir):
    if cfg.resume or cfg.resume_ckpt_fpath is not None:
        if cfg.resume_ckpt_fpath is not None:
            resume_ckpt_fpath = str(cfg.resume_ckpt_fpath)
        else:
            resume_ckpt_fpath = os.path.join(exp_dir, "fit", f"epoch_{cfg.resume_epoch}.ckpt") if cfg.resume else None
        resume_ckpt_fpath = os.path.abspath(resume_ckpt_fpath)
    else:
        resume_ckpt_fpath = None

    return resume_ckpt_fpath
