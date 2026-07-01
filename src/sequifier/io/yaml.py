import numpy
import yaml
from pydantic import BaseModel

from sequifier.config.train_config import (
    DotDict,
    ModelSpecModel,
    TrainingSpecModel,
    TrainModel,
)
from sequifier.helpers import ModelWindowView, StoredWindowLayout


def represent_sequifier_object(dumper, data):
    """Represent sequifier config objects as plain YAML mappings."""
    return dumper.represent_dict(data.__dict__)


def represent_dot_dict(dumper, data):
    """Represent DotDict as a plain YAML mapping."""
    return dumper.represent_dict(dict(data))


def represent_numpy_float(dumper, data):
    """Represent NumPy floats as YAML floats."""
    return dumper.represent_float(float(data))


def represent_numpy_int(dumper, data):
    """Represent NumPy integers as YAML integers."""
    return dumper.represent_int(int(data))


class TrainModelDumper(yaml.Dumper):
    """YAML dumper for sequifier config objects."""

    def increase_indent(self, flow=False, indentless=False):
        """Indent block sequences."""
        return super(TrainModelDumper, self).increase_indent(flow, False)


TrainModelDumper.add_representer(TrainModel, represent_sequifier_object)
TrainModelDumper.add_representer(ModelSpecModel, represent_sequifier_object)
TrainModelDumper.add_representer(TrainingSpecModel, represent_sequifier_object)
TrainModelDumper.add_representer(StoredWindowLayout, represent_sequifier_object)
TrainModelDumper.add_representer(ModelWindowView, represent_sequifier_object)
TrainModelDumper.add_multi_representer(BaseModel, represent_sequifier_object)
TrainModelDumper.add_representer(DotDict, represent_dot_dict)
TrainModelDumper.add_representer(numpy.float64, represent_numpy_float)
TrainModelDumper.add_representer(
    numpy.float32, represent_numpy_float
)  # Add for other numpy float types if necessary
TrainModelDumper.add_representer(numpy.int64, represent_numpy_int)
TrainModelDumper.add_representer(
    numpy.int32, represent_numpy_int
)  # Add for other numpy int types if necessary
