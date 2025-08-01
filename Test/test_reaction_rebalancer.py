import unittest
import logging
from synkit.IO import load_database
from synrbl import ReactionRebalancer, RebalanceConfig


class TestReactionRebalancer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load test data once for all tests
        cls.data = load_database("./Data/Testcase/testcase.json.gz")

    def test_rebalance_basic(self):
        # Basic pipeline test with logging disabled
        cfg = RebalanceConfig(id_col="id", enable_logging=False)
        rr = ReactionRebalancer(config=cfg)
        results = rr.rebalance(self.data, keep_extra=False)
        self.assertIsInstance(results, list)
        # Verify essential keys and non-null ids
        for entry in results:
            self.assertIn("id", entry)
            self.assertIn("reactions", entry)
            self.assertIsNotNone(entry["id"])

    def test_disable_logging(self):
        # Test global logging disable
        test_logger = logging.getLogger("test_logger")
        cfg = RebalanceConfig(enable_logging=False)
        _ = ReactionRebalancer(config=cfg, user_logger=test_logger)
        # logging.disable should be set to CRITICAL
        self.assertEqual(logging.root.manager.disable, logging.CRITICAL)

    def test_custom_logger(self):
        # Ensure custom logger is respected
        custom_logger = logging.getLogger("custom")
        custom_logger.disabled = False
        cfg = RebalanceConfig(enable_logging=True)
        rr = ReactionRebalancer(config=cfg, user_logger=custom_logger)
        self.assertIs(rr.logger, custom_logger)
        self.assertFalse(rr.logger.disabled)

    def test_raise_on_error(self):
        # Unsupported input should raise ValueError
        cfg = RebalanceConfig(id_col="id", raise_on_error=True)
        rr = ReactionRebalancer(config=cfg)
        with self.assertRaises(ValueError):
            rr.rebalance(123)

    def test_internal_id_preserved(self):
        # Internal 'R-id' should map back to external id_col
        cfg = RebalanceConfig(id_col="id")
        rr = ReactionRebalancer(config=cfg)
        results = rr.rebalance(self.data, keep_extra=True)
        for orig, res in zip(self.data, results):
            self.assertEqual(res["id"], res.get("R-id"))


if __name__ == "__main__":
    unittest.main()
