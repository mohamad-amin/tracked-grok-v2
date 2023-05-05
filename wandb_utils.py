import os
import time
import wandb
import shortuuid
import subprocess


def init_wandb(config, save_dir, specific_tag):

    tags = [
        config['data']['name'],
        config['model']['name'],
        config['train']['criterion'],
        specific_tag
    ]

    # git stuff
    try:
        commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
        gitdiff = subprocess.check_output(["git", "diff", "--", "':!notebooks'"]).decode()
    except:
        commit_sha = "n/a"
        gitdiff = ""

    wandb_args = {
        'project': 'tracked-grokking-v2',
        'tags': tags,
        'name': time.strftime('%Y%m%d-%H%M%S-') + str(shortuuid.uuid()),
        'config': {
            **config,
            "save_dir": save_dir,
            "commit": commit_sha,
            "gitdiff": gitdiff
        }
    }

    wandb.init(**wandb_args, sync_tensorboard=True)
