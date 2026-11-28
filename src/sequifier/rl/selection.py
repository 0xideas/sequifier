"""Comparable policy evaluations and durable best candidates."""

from dataclasses import dataclass
from uuid import uuid4

import torch

from sequifier.artifacts.model_artifact import read_execution_artifact
from sequifier.artifacts.phase import PhaseArtifacts, PhaseCommit, verify_artifact
from sequifier.artifacts.reports import HandoffCandidate
from sequifier.artifacts.state_dict import weight_fingerprint
from sequifier.config.policy_plan import DPOPhaseConfig, GRPOPhaseConfig, PolicyPhase
from sequifier.fingerprints import file_checksum, fingerprint
from sequifier.io.atomic import atomic_save


@dataclass(frozen=True)
class PolicyCandidate(HandoffCandidate):
    phase_index: int | None = None


def policy_evaluation_identity(
    phase: PolicyPhase,
    representation: str,
    execution: dict,
    *,
    validation,
    adapter,
    reference: str | None,
) -> str:
    evaluation = phase.evaluation.model_dump(
        mode="json", exclude={"cadence", "before_phase", "on_snapshot", "sources"}
    )
    contract = {
        "representation": representation,
        "execution": execution,
        "evaluation": evaluation,
    }
    if phase.evaluation_environment is not None:
        # PPO and GRPO use the same rollout evaluator. Only GRPO's score monitor
        # rescales its reported return.
        contract["adapter"] = adapter
        if (
            isinstance(phase, GRPOPhaseConfig)
            and phase.evaluation.monitor.metric == "score_mean"
        ):
            contract["score_scale"] = phase.reward_scale
    else:
        contract.update(algorithm=phase.algorithm, validation=validation)
        if isinstance(phase, DPOPhaseConfig):
            contract.update(
                reference=reference,
                beta=phase.beta,
                chunk_size=phase.scoring_chunk_size,
            )
    return fingerprint(contract)


def load_candidate(
    commit: PhaseCommit | None, evaluation_identity: str
) -> PolicyCandidate:
    if commit is None:
        return PolicyCandidate()
    selection = commit.sidecars.get("policy_selection")
    if selection is None or selection["evaluation"] != evaluation_identity:
        return PolicyCandidate()
    if identity := selection["weights"]:
        verify_artifact(
            identity["path"], identity["sha256"], "Committed policy candidate changed"
        )
        weights = torch.load(identity["path"], map_location="cpu", weights_only=True)
    else:
        commit.verify_model()
        weights = read_execution_artifact(commit.model_path).model_state_dict
    return PolicyCandidate(
        metric=selection["metric"],
        transaction=selection["transaction"],
        weights=weights,
        phase_index=selection["phase_index"],
    )


def stage_candidate(
    artifacts: PhaseArtifacts,
    candidate: PolicyCandidate,
    evaluation_identity: str,
    selected_weight_hash: str,
) -> dict | None:
    if candidate.metric is None:
        return None
    weights = candidate.require_weights()
    identity = None
    if weight_fingerprint(weights) != selected_weight_hash:
        path = artifacts.commits / f"{uuid4().hex}.best.pt"
        atomic_save(path, weights)
        identity = {"path": str(path), "sha256": file_checksum(path)}
    return {
        "evaluation": evaluation_identity,
        "metric": candidate.metric,
        "transaction": candidate.transaction,
        "phase_index": candidate.phase_index,
        "weights": identity,
    }
