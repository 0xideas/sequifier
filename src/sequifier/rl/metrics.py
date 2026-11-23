"""Policy observations in the native supervised metric tables."""

import math

from sequifier.training.metrics import StructuredMetricWriters


class PolicyMetrics:
    def __init__(self, config, identity):
        self.writer = StructuredMetricWriters(
            config.project_root,
            config.model_name,
            0,
        )
        self.run_id = identity.run_id
        self.session_id = identity.session_id
        self.pending = None

    def record(
        self,
        session,
        values,
        *,
        elapsed_seconds,
        validation=False,
        primary=None,
        initial=False,
        final=False,
    ):
        context = session.context
        if not validation:
            self.pending = (session, values, elapsed_seconds)
        if (
            not validation
            and not final
            and not session.stopping.triggered
            and session.transaction
            % (context.run.parent.run.logging.log_interval_iterations or 10)
        ):
            return
        scalars = {
            name: float(value)
            for name, value in values.items()
            if isinstance(value, (float, int))
            and name != "initial"
            and math.isfinite(value)
        }
        if not scalars:
            return
        if primary is None:
            primary = next(
                (
                    name
                    for name in (
                        "training_action_nll",
                        "actor_loss",
                        "training_dpo_loss",
                        "loss",
                        "policy_loss",
                    )
                    if name in scalars
                ),
                next(iter(scalars)),
            )
        self.writer.phase = context.phase.name
        self.writer.algorithm = context.phase.algorithm
        for role, (optimizer, _) in session.optimization.items():
            if role != "actor":
                scalars[f"{role}_learning_rate"] = float(
                    optimizer.param_groups[0]["lr"]
                )
        sources = context.phase.data_sources
        refs = sources.evaluation if validation else sources.training
        datasets = {ref.dataset for ref in refs}
        parts = {ref.part for ref in refs}
        self.writer.write_observation(
            dataset=next(iter(datasets)) if len(datasets) == 1 else None,
            part=next(iter(parts)) if len(datasets) == len(parts) == 1 else None,
            validation=validation,
            run_id=self.run_id,
            session_id=self.session_id,
            epoch=session.epoch,
            iteration=session.transaction,
            batches_total=session.iteration_budget,
            global_step=sum(session.steps.values()),
            values=scalars,
            primary=primary,
            learning_rate=float(session.optimization["actor"][0].param_groups[0]["lr"]),
            elapsed_seconds=elapsed_seconds,
            initial=initial,
        )
        if not validation:
            self.pending = None

    def flush(self, session):
        if self.pending is not None and self.pending[0] is session:
            _, values, elapsed_seconds = self.pending
            self.record(session, values, elapsed_seconds=elapsed_seconds, final=True)
