import os

from ....utils import import_modules


RM_FACTORY = {}

def RMFactory(reward_name):
    model = None
    for name in RM_FACTORY.keys():
        if name == reward_name.lower():
            model = RM_FACTORY[name]
    assert model, f"{reward_name} is not registered"
    return model


def register_rm(name):
    def register_rm_cls_or_func(cls_or_func):
        if name in RM_FACTORY:
            return RM_FACTORY[name]
        RM_FACTORY[name] = cls_or_func
        return cls_or_func
    return register_rm_cls_or_func


# automatically import any Python files in the models/ directory
models_dir = os.path.dirname(__file__)
import_modules(models_dir, "rllava.model.modules.rm")
