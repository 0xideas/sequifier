"""Adapt resolved phases to native execution and exact recovery."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from sequifier.artifacts.dependencies import (
    DependencyManifest,
    add_initialization,
    supervised_dependencies,
)
from sequifier.artifacts.phase import PhaseArtifacts
from sequifier.config.naming import supervised_phase_seed

if TYPE_CHECKING:
    from sequifier.rl.preparation import PolicyGroupPreparation
    from sequifier.rl.prepared_inputs import PreparedTrainingInputs
    from sequifier.rl.run_config import ResolvedPolicyRun
    from sequifier.training.runtime import PreparedSupervisedGroup


@dataclass
class SupervisedPhaseGroup:
    log_model_name: str
    index: int
    artifacts: PhaseArtifacts
    compiled: PreparedSupervisedGroup
    dependencies: DependencyManifest

    @property
    def phases(self):
        return self.compiled.plan.phases

    @property
    def config(self):
        return self.compiled.config

    def preflight(self, saved, committed_reference_count, *, finished=False):
        return self.dependencies, -1

    def release(self):
        self.compiled.parts.clear()

    def run(self, incoming, resuming, identity, publication, revision):
        from sequifier.runtime.context import ExecutionEnvironment
        from sequifier.training.supervised import execute_supervised

        config = self.config
        config.run.recovery.resume.policy = "required" if resuming else "never"
        add_initialization(self.dependencies, config, incoming)
        environment = ExecutionEnvironment(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(config.device),
            distributed=False,
        )
        execute_supervised(
            self.compiled,
            environment,
            initial_artifact=incoming,
            phase_offset=self.index,
            log_model_name=self.log_model_name,
            identity=identity,
            publication=publication,
            backbone_revision=revision,
            dependencies=self.dependencies,
        )


@dataclass
class PolicyPhaseGroup:
    artifacts: PhaseArtifacts
    config: ResolvedPolicyRun
    inputs: PreparedTrainingInputs
    preparation: PolicyGroupPreparation | None = None

    @property
    def phases(self):
        return self.config.phases

    @property
    def index(self):
        return self.config.phase_offset

    def preflight(self, saved, committed_reference_count, *, finished=False):
        from sequifier.rl.preparation import PolicyGroupPreparation

        self.preparation = PolicyGroupPreparation.build(
            self.config,
            self.inputs,
            group=True,
            known_adapters={} if finished else None,
            saved=saved,
            committed_reference_count=committed_reference_count,
        )
        last_data_phase = max(
            (
                i
                for i, data in enumerate(self.preparation.data, self.index)
                if data is not None
            ),
            default=-1,
        )
        return self.preparation.dependencies, last_data_phase

    def release(self):
        self.preparation = None

    def run(self, incoming, resuming, identity, publication, revision):
        from sequifier.rl.learner import execute_policy

        assert self.preparation is not None
        run = replace(
            self.config,
            initialize_from=incoming,
            resume=self.config.resume.model_copy(
                update={"policy": "required" if resuming else "never"}
            ),
        )
        execute_policy(
            run,
            self.inputs,
            identity=identity,
            publication=publication,
            preparation=self.preparation,
            backbone_revision=revision if self.index == 0 else None,
        )


def compile_phase_group(config, phases, index, inputs):
    from sequifier.config.training_resolution import policy_run_config
    from sequifier.training.runtime import PreparedSupervisedGroup

    phase = phases[0]
    artifacts = PhaseArtifacts.for_phase(
        config.outputs.phase_directory, index, phase.name
    )
    resume = config.run.recovery.resume.model_copy(
        update={"checkpoint_path": str(artifacts.recovery), "policy": "never"}
    )
    if phase.algorithm != "supervised":
        native = policy_run_config(config, phases, index=index, resume=resume)
        return PolicyPhaseGroup(artifacts, native, inputs)
    names = {
        ref.dataset
        for item in phases
        for ref in [*item.sources, *item.evaluation_sources]
    }
    datasets = {
        name: dataset
        for name, dataset in config.dataset_training.items()
        if name in names
    }
    seed = (
        phase.resolved_seed
        if phase.resolved_seed is not None
        else supervised_phase_seed(config.seed, index)
    )
    model_name = f"{config.model_name}-phase-{index:03d}"
    resolved = config.model_copy(
        update={
            "dataset_training": datasets,
            "training_plan": list(phases),
            "run": config.run.model_copy(
                update={
                    "model_name": model_name,
                    "reproducibility": config.run.reproducibility.model_copy(
                        update={"seed": seed}
                    ),
                    "recovery": config.run.recovery.model_copy(
                        update={"resume": resume}
                    ),
                }
            ),
            "initialize_from": None,
            "actor_initialization": "attach",
            "curriculum_training": bool(phase.curriculum),
            "curriculum_column": phase.curriculum.column if phase.curriculum else None,
        }
    )
    compiled = PreparedSupervisedGroup.compile(resolved, validate_sources=True)
    return SupervisedPhaseGroup(
        config.model_name,
        index,
        artifacts,
        compiled,
        supervised_dependencies(compiled.config),
    )
