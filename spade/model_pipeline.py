from torch import nn, no_grad, Tensor, cuda
import torch
import importlib
import inspect
from functools import lru_cache
import numpy as np
import json


def unsafe_hash(d):
    return hash(json.dumps(d))


MODELS = dict()

# Example config
# pipelines:
# - model: ModelA
#   num: 4
# - model: ModelB
#   num: 8
# - model: torch.nn.Linear
#   in_features: 8
#   out_features: 8


def parse_pipeline_config(pipelines):
    stages = []
    # assert 'stage' in pipelines[0], "Pipeline must starts with a stage"
    # stage_name = pipelines[0]['stage']
    current_stage = dict(name=None, models=[])
    for model_config in pipelines:
        if 'stage' in model_config:
            if current_stage['name'] is not None:
                stages.append(current_stage)
            current_stage = dict(name=model_config['stage'], models=[])
        else:
            current_stage['models'].append(model_config)

    stages.append(current_stage)

    return stages


class Pipeline(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.device = config.get(
            'device',
            'cuda' if cuda.is_available() else 'cpu',
        )

        self.preprocess = self.get_modules(config['pipeline']['preprocess'])
        self.postprocess = self.get_modules(config['pipeline']['postprocess'])
        self.core = self.get_modules(config['pipeline']['core'])

        self.preprocessed_data = dict()

    def get_module(self, name):
        if name in MODELS:
            return MODELS[name]
        else:
            parts = name.split(".")
            attr = parts[-1]
            module_ = '.'.join(parts[:-1])
            module_ = importlib.import_module(module_)
            return getattr(module_, attr)

    def get_modules(self, pipeline_config):
        modules = nn.ModuleList()
        for model_config in pipeline_config:
            Model = self.get_module(model_config["model"])
            output_mapping = model_config.get("output_mapping", None)
            input_mapping = model_config.get("input_mapping", None)
            model_config = {
                k: v
                for (k, v) in model_config.items()
                if k not in ["model", "input_mapping", "output_mapping"]
            }
            model = Model(**model_config)
            model.output_mapping = output_mapping
            model.input_mapping = input_mapping
            modules.append(model)
        return modules

    @lru_cache
    def get_input_names(self, forward):
        return inspect.signature(forward).parameters.keys()

    def forward_stage(self, stage, states):
        for model in stage:
            input_keys = self.get_input_names(model.forward)
            if model.input_mapping is not None:
                inputs = {
                    k: states[model.input_mapping[k]]
                    for k in input_keys
                }
            else:
                inputs = {k: states[k] for k in input_keys}
            outputs = model(**inputs)
            if model.output_mapping is not None:
                outputs = {
                    model.output_mapping[k]: v
                    for (k, v) in outputs.items()
                }
            states.update(outputs)
        return states

    def forward(self, data_id, **states):
        if data_id not in self.preprocessed_data:
            self.preprocessed_data[data_id] = self.forward_stage(
                self.preprocess, states)
        states = self.preprocessed_data[data_id]

        # Convert to tensor and move to device
        states = states.copy()
        for (k, v) in states.items():
            if isinstance(v, np.ndarray):
                v = torch.tensor(v)
            if isinstance(v, Tensor):
                states[k] = v.to(self.device)
                if states[k].dtype == torch.double:
                    states[k] = states[k].float()

        states = self.forward_stage(self.core, states)

        # Move back to CPU
        for (k, v) in states.items():
            if isinstance(v, Tensor):
                states[k] = v.to('cpu')

        states = self.forward_stage(self.postprocess, states)

        return states
