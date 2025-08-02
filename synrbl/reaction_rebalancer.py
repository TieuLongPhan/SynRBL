"""
reaction_rebalancer.py

Provides ReactionRebalancer for rebalancing chemical reactions pipeline: standardization,
balancing (which includes its own post-processing), neutralization, and deionization.
Uses internal 'R-id' copied from external id_col when they differ.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from synrbl import Balancer
from synkit.IO.debug import setup_logging
from synkit.Chem.Reaction.standardize import Standardize
from synkit.Chem.Reaction.deionize import Deionize
from synkit.Chem.Reaction.neutralize import Neutralize

# Initialize module-level logger
logger = setup_logging()


@dataclass(frozen=True)
class RebalanceConfig:
    """
    Configuration parameters for the reaction rebalancing pipeline.

    :param reaction_col: Column name for reaction strings in input data.
    :type reaction_col: str
    :param id_col: Column name for reaction identifiers (external).
    :type id_col: str
    :param n_jobs: Number of parallel jobs for processing.
    :type n_jobs: int
    :param batch_size: Batch size for the Balancer.
    :type batch_size: int
    :param raise_on_error: Whether to raise exceptions or log and continue.
    :type raise_on_error: bool
    :param enable_logging: Whether to enable logging output.
    :type enable_logging: bool
    :param use_default_reduction: Passed to Balancer to control reduction behavior.
    :type use_default_reduction: bool
    """

    reaction_col: str = "reactions"
    id_col: str = "R-id"
    n_jobs: int = 1
    batch_size: int = 1000
    raise_on_error: bool = False
    enable_logging: bool = True
    use_default_reduction: bool = False


class ReactionRebalancer:
    """
    Orchestrates the pipeline for rebalancing chemical reactions:
    standardization, balancing (including its internal post-processing),
    neutralization, and deionization.
    Uses an internal 'R-id' field, initially copied from the external id_col,
    and writes back at the end. Logging can be toggled via config.

    :param config: Configuration for the rebalancer.
    :type config: RebalanceConfig
    :param user_logger: Logger for recording pipeline events.
    :type user_logger: logging.Logger
    :raises TypeError: If config is not a RebalanceConfig or logger is not a Logger.
    """

    INTERNAL_ID: str = "R-id"

    def __init__(
        self,
        config: Optional[RebalanceConfig] = None,
        user_logger: Optional[logging.Logger] = None,
    ):
        # Validate inputs
        if config is not None and not isinstance(config, RebalanceConfig):
            raise TypeError(f"config must be RebalanceConfig, got {type(config)}")
        if user_logger is not None and not isinstance(user_logger, logging.Logger):
            raise TypeError(f"logger must be logging.Logger, got {type(user_logger)}")

        self.config = config or RebalanceConfig()
        # Use provided logger or default module-level logger
        self.logger = user_logger if user_logger is not None else logger
        # Disable logging if configured
        if not self.config.enable_logging:
            logging.disable(logging.CRITICAL)
        self.standardizer = Standardize()

    def __repr__(self) -> str:
        """
        :returns: Human-readable representation of the instance.
        :rtype: str
        """
        return f"{self.__class__.__name__}(config={self.config})"

    @staticmethod
    def describe() -> None:
        """
        Prints usage examples for the ReactionRebalancer.

        :example:
        >>> ReactionRebalancer.describe()
        """
        print(
            "Usage examples for ReactionRebalancer:\n"
            "  rr = ReactionRebalancer()\n"
            "  result = rr.rebalance(data_frame_or_list)"
        )

    def rebalance(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]], keep_extra: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Execute the full rebalancing pipeline on input data.

        :param data: Input data, DataFrame or list of dicts.
        :type data: Union[pd.DataFrame, List[Dict[str, Any]]]
        :param keep_extra: Retain intermediate fields if True.
        :type keep_extra: bool
        :returns: List of dicts with keys id_col and reaction_col (+ extras).
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If input type unsupported.
        """
        cfg = self.config
        ext_id, int_id, rxn_col = cfg.id_col, self.INTERNAL_ID, cfg.reaction_col
        self.logger.info("Starting rebalancing pipeline.")

        records = self._load_records(data)
        self._init_ids(records, ext_id, int_id)
        std_data = self._standardize_records(records, rxn_col)
        balanced = self._balance_reactions(std_data, rxn_col, int_id)
        restored = self._restore_internal_id(std_data, balanced)
        fixed = self._neutralize_deionize(restored, rxn_col)
        return self._extract_results(fixed, ext_id, int_id, rxn_col, keep_extra)

    def _load_records(
        self, data: Union[pd.DataFrame, List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        if isinstance(data, pd.DataFrame):
            self.logger.debug("Converted DataFrame to records, count=%d", len(data))
            return data.to_dict("records")
        if isinstance(data, list) and all(isinstance(x, dict) for x in data):
            return [x.copy() for x in data]
        msg = f"Unsupported data type: {type(data)}"
        self.logger.error(msg)
        raise ValueError(msg)

    def _init_ids(
        self, records: List[Dict[str, Any]], ext_id: str, int_id: str
    ) -> None:
        if ext_id != int_id:
            for entry in records:
                entry[int_id] = entry.get(ext_id)

    def _standardize_records(
        self, records: List[Dict[str, Any]], rxn_col: str
    ) -> List[Dict[str, Any]]:
        out = []
        self.logger.info("Standardizing reactions.")
        for e in records:
            raw = e.get(rxn_col)
            if not raw:
                self.logger.warning("Missing '%s'; skipping.", rxn_col)
                e["std_rxn"] = None
                continue
            try:
                std = self.standardizer.fit(raw, remove_aam=True)
                e["std_rxn"], e[rxn_col] = std, std
                out.append(e)
            except Exception:
                self.logger.exception("Std failed: %s", raw)
        if not out:
            self.logger.error("No valid standardized reactions.")
            if self.config.raise_on_error:
                raise RuntimeError("Zero valid entries.")
        return out

    def _balance_reactions(
        self, data: List[Dict[str, Any]], rxn_col: str, int_id: str
    ) -> List[Dict[str, Any]]:
        self.logger.info("Balancing %d reactions.", len(data))
        try:
            balancer = Balancer(
                reaction_col=rxn_col,
                id_col=int_id,
                n_jobs=self.config.n_jobs,
                batch_size=self.config.batch_size,
                use_default_reduction=self.config.use_default_reduction,
            )
            return balancer.rebalance(reactions=data, output_dict=True)
        except Exception:
            self.logger.exception("Balancer failed.")
            if self.config.raise_on_error:
                raise
            return []

    def _neutralize_deionize(
        self, data: List[Dict[str, Any]], rxn_col: str
    ) -> List[Dict[str, Any]]:
        self.logger.info("Neutralization.")
        try:
            data = Neutralize.parallel_fix_unbalanced_charge(
                data, rxn_col, self.config.n_jobs
            )
        except Exception:
            self.logger.exception("Neutralization failed.")
            if self.config.raise_on_error:
                raise
        self.logger.info("Deionization.")
        try:
            data = Deionize.apply_uncharge_smiles_to_reactions(
                data, Deionize.uncharge_smiles, n_jobs=1
            )
        except Exception:
            self.logger.exception("Deionization failed.")
            if self.config.raise_on_error:
                raise
        return data

    def _extract_results(
        self,
        data: List[Dict[str, Any]],
        ext_id: str,
        int_id: str,
        rxn_col: str,
        keep_extra: bool,
    ) -> List[Dict[str, Any]]:
        self.logger.info("Extracting results.")
        results = []
        for e in data:
            if ext_id != int_id:
                e[ext_id] = e.get(int_id)
            final = e.get("standardized_reactions", e.get(rxn_col))
            out = {ext_id: e.get(ext_id), rxn_col: final}
            if keep_extra:
                extras = {k: v for k, v in e.items() if k not in (ext_id, rxn_col)}
                out.update(extras)
            results.append(out)
        return results

    @staticmethod
    def _restore_internal_id(
        original_list: List[Dict[str, Any]], processed_list: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Restore internal 'R-id' based on std_rxn mapping.

        :param original_list: List of dicts with 'std_rxn' and INTERNAL_ID fields.
        :type original_list: List[Dict[str, Any]]
        :param processed_list: List of dicts with 'input_reaction' and INTERNAL_ID.
        :type processed_list: List[Dict[str, Any]]
        :returns: New list with consistent internal 'R-id'.
        :rtype: List[Dict[str, Any]]
        :raises ValueError: If input lists are not lists of dicts or
        if required keys missing.
        """
        if not all(isinstance(x, dict) for x in original_list + processed_list):
            raise ValueError(
                "Both original_list and processed_list must be lists of dicts."
            )

        int_id = ReactionRebalancer.INTERNAL_ID
        mapping: Dict[str, List[Dict[str, Any]]] = {}
        for entry in original_list:
            std_rxn = entry.get("std_rxn")
            if not std_rxn:
                continue
            mapping.setdefault(std_rxn, []).append(entry)

        out_list: List[Dict[str, Any]] = []
        std_obj = Standardize()
        for entry in processed_list:
            if "input_reaction" not in entry:
                raise ValueError(f"Processed entry missing 'input_reaction': {entry}")
            new_entry = entry.copy()
            input_raw = entry["input_reaction"]

            # First try a direct lookup (maybe already standardized)
            originals = mapping.get(input_raw, [])
            if not originals:
                # Fallback: standardize then lookup
                try:
                    standardized_input = std_obj.fit(input_raw, remove_aam=True)
                    originals = mapping.get(standardized_input, [])
                except Exception:
                    logger.exception(
                        "Standardization of input_reaction failed: %s", input_raw
                    )

            if originals:
                new_entry[int_id] = originals[0].get(int_id)
            out_list.append(new_entry)

        return out_list
