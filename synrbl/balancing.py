import copy
import logging
import pandas as pd

from synrbl.preprocess import preprocess
from synrbl.postprocess import Validator
from synrbl.rule_based import RuleBasedMethod
from synrbl.mcs_search import MCSSearch
from synrbl.SynMCSImputer.mcs_based_method import MCSBasedMethod
from synrbl.SynChemImputer.post_process import PostProcess
from synrbl.SynChemImputer.molecule_standardizer import MoleculeStandardizer
from synrbl.confidence_prediction import ConfidencePredictor
from synrbl.SynUtils.batching import Dataset, DataLoader, CacheManager

logger = logging.getLogger("synrbl")


def merge_stats(stats, new_stats):
    if stats is None:
        return
    stats_keys = list(stats.keys())
    new_stats_keys = list(new_stats.keys())
    for k in stats.keys():
        if k in new_stats_keys:
            stats[k] += new_stats[k]
    for k, v in new_stats.items():
        if k not in stats_keys:
            stats[k] = v


class Balancer:
    def __init__(
        self,
        id_col="id",
        reaction_col="reaction",
        confidence_threshold=0,
        n_jobs=-1,
        batch_size=None,
        cache=False,
        cache_dir: str | None = "./cache",
    ):
        self.__reaction_col = reaction_col
        self.__id_col = id_col
        self.__solved_col = "solved"
        self.__solved_by_col = "solved_by"
        self.__mcs_data_col = "mcs"
        self.__input_col = "input_reaction"
        self.__confidence_col = "confidence"
        self.__unbalance_col = "unbalance_col"
        self.__carbon_balance_col = "carbon_balance_check"
        self.__rules_col = "rules"
        self.__issue_col = "issue"
        self.__n_jobs = n_jobs

        self.batch_size = batch_size
        self.cache = cache
        self.cache_dir = cache_dir
        self.columns = [
            self.__input_col,
            reaction_col,
            self.__solved_col,
            self.__solved_by_col,
            self.__confidence_col,
            self.__rules_col,
            self.__issue_col,
        ]

        self.confidence_threshold = confidence_threshold
        self.input_validator = Validator(
            reaction_col,
            "input-balanced",
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )
        self.rb_validator = Validator(
            reaction_col,
            "rule-based",
            check_carbon_balance=False,
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )
        self.mcs_validator = Validator(
            reaction_col,
            "mcs-based",
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )

        self.rb_method = RuleBasedMethod(
            id_col, reaction_col, reaction_col, n_jobs=n_jobs
        )
        self.mcs_search = MCSSearch(
            id_col,
            solved_col=self.__solved_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            n_jobs=n_jobs,
        )
        self.mcs_method = MCSBasedMethod(
            reaction_col,
            reaction_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            rules_col=self.__rules_col,
            smiles_standardizer=[MoleculeStandardizer()],
        )
        self.post_processor = PostProcess(
            id_col=id_col, reaction_col=reaction_col, n_jobs=n_jobs, verbose=0
        )
        self.conf_predictor = ConfidencePredictor(
            reaction_col=reaction_col,
            solved_by_method="mcs-based",
            input_reaction_col=self.__input_col,
            confidence_col=self.__confidence_col,
            solved_col=self.__solved_col,
            solved_by_col=self.__solved_by_col,
            issue_col=self.__issue_col,
            mcs_col=self.__mcs_data_col,
        )

    @property
    def n_jobs(self):
        return self.__n_jobs

    @n_jobs.setter
    def n_jobs(self, value):
        self.__n_jobs = value

        self.input_validator.n_jobs = value
        self.rb_validator.n_jobs = value
        self.mcs_validator.n_jobs = value
        self.rb_method.n_jobs = value
        self.mcs_search.n_jobs = value
        self.post_processor.n_jobs = value

    def __post_process(self, reactions):
        key_index_map = {item[self.__id_col]: idx for idx, item in enumerate(reactions)}
        pp_data = [
            r
            for r in reactions
            if self.__solved_by_col in r.keys()
            and r[self.__solved_by_col] != "input-balanced"
        ]
        pp_results = self.post_processor.fit(pp_data)
        for pp_result in pp_results:
            if (
                pp_result["label"] != "unspecified"
                and "curated_reaction" in pp_result.keys()
            ):
                idx = key_index_map[pp_result[self.__id_col]]
                reactions[idx][self.__reaction_col] = pp_result["curated_reaction"]

    def __run_pipeline(self, reactions, stats=None):
        if stats is not None:
            stats["reaction_cnt"] = len(reactions)
        reactions = preprocess(
            reactions,
            self.__reaction_col,
            self.__id_col,
            self.__solved_col,
            self.__input_col,
        )
        rxn_cnt = len(reactions)
        self.input_validator.check(reactions)

        logger.info("Run rule-based method.")
        self.rb_method.run(reactions, stats=stats)
        self.rb_validator.check(reactions, override_unsolved=True)

        self.mcs_search.find(reactions)

        logger.info("Impute missing compounds from MCS.")
        self.mcs_method.run(reactions, stats=stats)
        self.mcs_validator.check(reactions)  # update carbon balance
        logger.info(
            "Run rule-based method again to fix remaining non-carbon imbalance."
        )
        self.__post_process(reactions)
        self.rb_method.run(reactions)
        self.mcs_validator.check(reactions, override_unsolved=True)

        self.conf_predictor.predict(
            reactions, stats=stats, threshold=self.confidence_threshold
        )

        assert rxn_cnt == len(reactions)

        logger.info("DONE")

        return reactions

    def rebalance(
        self,
        reactions,
        output_dict=False,
        stats=None,
        batch_size=None,
        cache=None,
        cache_dir=None,
    ):
        dataset = None
        if isinstance(reactions, str):
            reactions = [reactions]
        if isinstance(reactions, list):
            reaction_data = []
            for r in reactions:
                if isinstance(r, str):
                    reaction_data.append({self.__reaction_col: r})
                elif isinstance(r, dict):
                    reaction_data.append(r)
                else:
                    raise ValueError(
                        "Expected (a list of) SMILES or a data dictionary. "
                        + "Found '{}' instead.".format(type(r))
                    )
            dataset = Dataset(reaction_data)
        if isinstance(reactions, Dataset):
            dataset = reactions
        if dataset is None:
            raise ValueError(
                "Invalid type '{}' of reactions. "
                + "Use a list of SMILES or a Dataset instead.".format(type(reactions))
            )

        batch_size = self.batch_size if batch_size is None else batch_size
        cache = self.cache if cache is None else cache
        cache_dir = self.cache_dir if cache_dir is None else cache_dir

        if batch_size is None:
            dataloader = iter([[e for e in dataset]])
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size)

        results = []
        failed_batches = []
        failed_cnt = 0
        cache_manager = None
        if cache:
            if cache_dir is None:
                raise ValueError(
                    "Undefined cache directory. "
                    + "Specify a directory with 'cache_dir' argument."
                )
            cache_manager = CacheManager(cache_dir=cache_dir)
        for batch_i, batch in enumerate(dataloader):
            batch_i = batch_i + 1
            if len(batch) == 0:
                continue
            if batch_size is not None:
                logger.info("Start Batch {} | Size: {}".format(batch_i, len(batch)))

            result = None
            batch_stats = {}
            cache_key = None
            if cache_manager:
                cache_key = cache_manager.get_hash_key(batch)
                if cache_manager.is_cached(cache_key):
                    logger.info("Load cached results. (Key: {})".format(cache_key[:8]))
                    cache_result = cache_manager.load_cache(cache_key)
                    result = cache_result.get("result", None)
                    batch_stats = cache_result.get("stats", None)

            if result is None or batch_stats is None:
                try:
                    result = self.__run_pipeline(copy.deepcopy(batch), batch_stats)
                    if cache_manager:
                        assert cache_key is not None
                        cache_manager.write_cache(
                            cache_key, {"stats": batch_stats, "result": result}
                        )
                        logger.info(
                            "Cached new results. (Key: {})".format(cache_key[:8])
                        )
                except Exception as e:
                    logger.error("Pipeline execution failed: {}".format(e))
                    failed_batches.append((batch_i, e))

            if result is not None:
                results.extend(result)
                merge_stats(stats, batch_stats)

            if batch_size is not None:
                logger.info("Completed batch {}".format(batch_i))

        if batch_size is not None and len(failed_batches) > 0:
            logger.info("Failed batches:")
            for i, e in failed_batches:
                logger.info("  [{}] Exception: {}".format(i, e))
            logger.info(
                "Computation failed for {} batches with a total of {} reactions. "
                + "These are not part of the result.".format(
                    len(failed_batches), failed_cnt
                )
            )

        if output_dict:
            output = []
            for r in results:
                output.append({k: v for k, v in r.items() if k in self.columns})
            return output
        else:
            return [r[self.__reaction_col] for r in results]
