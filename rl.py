"""Launch the OFFICIAL FastRL Hydra/Ray RL entrypoint, not a rewritten trainer."""
import argparse
from tlt_reflex.runtime import configure,require_runtime


def compose_config(overrides):
    from hydra import compose,initialize_config_dir
    from tlt_reflex.runtime import FAStrl
    with initialize_config_dir(config_dir=str(FAStrl/'verl/trainer/config'),version_base=None):
        return compose(config_name='fastrl_trainer',overrides=overrides)


def validate_spot_trainer(cfg):
    if cfg.speculative.train.enable_drafter_training:
        raise ValueError('Current mode is TLT adaptive speculative rollout + fixed pretrained EAGLE3, NOT full Spot-Trainer TLT. Pinned FSDP factory/capture/loss/weight sync are EAGLE1; EAGLE3 Spot Trainer is unsupported, keep speculative.train.enable_drafter_training=false.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['tlt','tlt_opd_reflex'],required=True)
    p.add_argument('--validate-config',action='store_true')
    args,overrides=p.parse_known_args()
    configure(args.method)
    cfg=compose_config(overrides)
    validate_spot_trainer(cfg)  # validate effective Hydra booleans, not text spelling
    if args.validate_config:
        # Compose the REAL upstream schema, without importing Ray/CUDA/models.
        from omegaconf import OmegaConf
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    require_runtime(rl=True)
    import sys
    from verl.trainer.main_fastrl import main as upstream_main
    sys.argv=['verl.trainer.main_fastrl']+overrides
    upstream_main()


if __name__=='__main__':main()
