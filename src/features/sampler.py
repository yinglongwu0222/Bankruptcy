"""
Sampler factory.
"""

import logging
from typing import Optional

from imblearn.base import BaseSampler
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss

logger = logging.getLogger(__name__)

class SamplerFactory:
    """
    Helper to get sampler instances.
    """

    @staticmethod
    def get_sampler(method_name: Optional[str], random_state: int = 42) -> Optional[BaseSampler]:
        """
        Retrieves the specified imbalanced-learn sampler.

        Args:
            method_name (str): The name of the sampling algorithm.
            random_state (int): Seed for deterministic synthesis.

        Returns:
            BaseSampler: The instantiated sampler, or None if no sampling is requested.
        """
        if not method_name or method_name.strip().lower() == "none":
            logger.info("No resampling technique applied (Baseline).")
            return None

        normalized_name = method_name.strip().upper()
        logger.info(f"Configuring sampling strategy: {normalized_name}")

        # Dictionary mapping for O(1) retrieval
        samplers = {
            "RANDOM": RandomOverSampler(random_state=random_state),
            "SMOTE": SMOTE(random_state=random_state),
            "ADASYN": ADASYN(random_state=random_state),
            "NEARMISS": NearMiss(),
            "SMOTEENN": SMOTEENN(random_state=random_state),
        }

        if normalized_name not in samplers:
            error_msg = f"Unsupported sampling method: {method_name}. Available options: {list(samplers.keys())}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return samplers[normalized_name]