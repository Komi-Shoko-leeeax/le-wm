"""Precision-aware planning cost adapter for LeWM."""

from stable_worldmodel.planning import ShootingCostEvaluator

from .quantization import autocast_context


class PrecisionAwareShootingCostEvaluator(ShootingCostEvaluator):
    """Run LeWM rollout/cost evaluation under the configured precision."""

    def get_cost(self, info_dict, action_candidates):
        with autocast_context(self.model):
            return super().get_cost(info_dict, action_candidates)
