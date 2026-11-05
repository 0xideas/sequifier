import pytest
from torch import nn

from sequifier.config.freezing_config import LayerFreezingConfigFields
from sequifier.model.parameter_catalog import (
    ParameterCatalog,
    semantic_optimizer_groups,
)
from sequifier.training.runtime import _policy_parameter_ids


@pytest.mark.parametrize(
    "normalization",
    [
        lambda: nn.GroupNorm(2, 4),
        lambda: nn.BatchNorm1d(4),
        lambda: nn.BatchNorm2d(4),
        lambda: nn.BatchNorm3d(4),
    ],
)
def test_normalization_classification_freezing_and_optimizer_routing(normalization):
    model = nn.Module()
    model.ingestion = nn.Sequential(normalization(), nn.Linear(4, 4))
    model.backbone = nn.LayerNorm(4)
    model.backbone.weight._sequifier_depth_parameter = True

    catalog = ParameterCatalog(model)
    ingestion_normalization = catalog.select(
        component="ingestion", semantic_group="normalization"
    )
    backbone_normalization = catalog.select(
        component="backbone", semantic_group="normalization"
    )
    assert {item.parameter_kind for item in ingestion_normalization} == {
        "weight",
        "bias",
    }
    assert {item.parameter_kind for item in backbone_normalization} == {
        "weight",
        "bias",
    }

    normalization_ids = {
        id(catalog.parameter(item.parameter_id)) for item in ingestion_normalization
    }
    frozen = _policy_parameter_ids(
        model.ingestion,
        LayerFreezingConfigFields(freeze=["normalization"]),
        "ingestion",
    )
    assert frozen == normalization_ids

    optimizer_groups = {
        group["group_id"]: {id(parameter) for parameter in group["params"]}
        for group in semantic_optimizer_groups(catalog)
    }
    assert optimizer_groups["ingestion.normalization"] == normalization_ids
    assert optimizer_groups["backbone.normalization"] == {
        id(catalog.parameter(item.parameter_id)) for item in backbone_normalization
    }
    assert "ingestion.free_parameter" not in optimizer_groups
