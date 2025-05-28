import numpy
import yaml

from sequifier.config.train_config import (
    DotDict,
    ModelSpecModel,
    TrainingSpecModel,
    TrainModel,
)


def represent_sequifier_object(dumper, data):
    """
    Represents objects from 'sequifier.config.train_config' (like TrainModel,
    ModelSpecModel, TrainingSpecModel) as a simple YAML mapping,
    using the object's __dict__. This effectively removes the
    !!python/object tag and the explicit '__dict__:', '__fields_set__:' keys.
    """
    # We assume the object's __dict__ contains the attributes to be serialized.
    # If these objects are Pydantic models, using data.model_dump(mode='python')
    # would be more robust if available.
    return dumper.represent_dict(data.__dict__)


def represent_dot_dict(dumper, data):
    """
    Represents DotDict objects as a simple YAML mapping.
    The original output showed a 'dictitems' attribute. If your DotDict
    is essentially a dictionary, this will work.
    """
    # If DotDict has a specific attribute like 'dictitems' that holds the actual dict:
    # return dumper.represent_dict(data.dictitems)
    # If DotDict is a subclass of dict or dict-like:
    return dumper.represent_dict(dict(data))


def represent_numpy_float(dumper, data):
    """
    Represents numpy.float64 (and similar numpy floats) as standard YAML floats.
    """
    return dumper.represent_float(float(data))


def represent_numpy_int(dumper, data):
    """
    Represents numpy.int64 (and similar numpy integers) as standard YAML integers.
    """
    return dumper.represent_int(int(data))


class TrainModelDumper(yaml.Dumper):
    # You can add more customizations here if needed, like indent width.
    def increase_indent(self, flow=False, indentless=False):
        return super(TrainModelDumper, self).increase_indent(flow, False)


TrainModelDumper.add_representer(TrainModel, represent_sequifier_object)
TrainModelDumper.add_representer(ModelSpecModel, represent_sequifier_object)
TrainModelDumper.add_representer(TrainingSpecModel, represent_sequifier_object)
TrainModelDumper.add_representer(DotDict, represent_dot_dict)
TrainModelDumper.add_representer(numpy.float64, represent_numpy_float)
TrainModelDumper.add_representer(
    numpy.float32, represent_numpy_float
)  # Add for other numpy float types if necessary
TrainModelDumper.add_representer(numpy.int64, represent_numpy_int)
TrainModelDumper.add_representer(
    numpy.int32, represent_numpy_int
)  # Add for other numpy int types if necessary
