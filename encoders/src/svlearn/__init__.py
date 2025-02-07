import os

root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

CONFIG_PATH = f"{root_path}/fine-tuning-config.yaml"

__all__ = ['CONFIG_PATH']