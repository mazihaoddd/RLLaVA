import os
from typing import Dict

from .base import *
from ...utils import import_modules


TEMPLATE_FACTORY: Dict[str, Template] = {}

def TemplateFactory(version: str):
    template = TEMPLATE_FACTORY.get(version)
    if template is None:
        available = sorted(TEMPLATE_FACTORY.keys())
        raise ValueError(
            f"Template '{version}' is not implemented. Available templates: {available}"
        )
    return template

def available_templates() -> List[str]:
    return sorted(TEMPLATE_FACTORY.keys())

def register_template(name):
    def register_template_cls(cls):
        if name in TEMPLATE_FACTORY:
            return TEMPLATE_FACTORY[name]

        TEMPLATE_FACTORY[name] = cls
        return cls

    return register_template_cls


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "rllava.data.template")
