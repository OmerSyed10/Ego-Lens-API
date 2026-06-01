"""
egolens — Config-driven egocentric video QC pipeline.

Quick start
-----------
    from egolens.config   import load_config
    from egolens.pipeline import EgoLensPipeline
    from egolens.scorer   import score
    from egolens.reporter import print_report

    cfg    = load_config()                   # reads egolens/config.yaml
    result = EgoLensPipeline(cfg).run("video.mp4")
    scored = score(result, cfg)
    print_report(result, scored)

CLI
---
    python -m egolens video.mp4 [--config config.yaml]
"""

__version__ = "0.1.0"
