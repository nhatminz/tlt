"""Launch the OFFICIAL FastRL Hydra/Ray RL entrypoint, not a rewritten trainer."""
import argparse
from tlt_reflex.runtime import configure,require_runtime


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method',choices=['tlt','tlt_reflex'],required=True)
    p.add_argument('--validate-config',action='store_true')
    args,overrides=p.parse_known_args()
    configure(args.method)
    if args.validate_config:
        # Compose the REAL upstream schema, without importing Ray/CUDA/models.
        from hydra import compose,initialize_config_dir
        from omegaconf import OmegaConf
        from tlt_reflex.runtime import FAStrl
        with initialize_config_dir(config_dir=str(FAStrl/'verl/trainer/config'),version_base=None):
            cfg=compose(config_name='fastrl_trainer',overrides=overrides)
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    # The upstream background trainer factory selects EAGLE1 only. Do not
    # pretend it supports EAGLE3 or silently use a different architecture.
    if any(x=='speculative.train.enable_drafter_training=true' for x in overrides):
        raise ValueError('pinned upstream background trainer selects EAGLE1; EAGLE3 online training is not implemented by this plugin. Keep upstream default false.')
    require_runtime(rl=True)
    import sys
    from verl.trainer.main_fastrl import main as upstream_main
    sys.argv=['verl.trainer.main_fastrl']+overrides
    upstream_main()


if __name__=='__main__':main()
