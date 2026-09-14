import csv
import math
import tempfile
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from run_utils import MetricsLogger
from sts_eval import TokenizedSTSPairs, evaluate_tokenized_sts


class _LookupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer(
            "vectors",
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [math.sqrt(0.5), math.sqrt(0.5)],
                ]
            ),
        )


def _lookup_encode(model, tokens, pad_mask, cfg):
    del pad_mask, cfg
    return F.normalize(model.vectors[tokens[:, 0]], dim=-1), None


class STSEvaluationTests(unittest.TestCase):
    def test_correlations_and_training_mode_are_preserved(self):
        pairs = TokenizedSTSPairs(
            name="synthetic",
            split="validation",
            sentence1=[[1], [1], [1]],
            sentence2=[[1], [2], [3]],
            scores=[1.0, 0.0, math.sqrt(0.5)],
            pad_id=0,
        )
        model = _LookupModel()
        model.train()
        result = evaluate_tokenized_sts(
            pairs,
            model,
            cfg=None,
            device="cpu",
            batch_size=2,
            encode_fn=_lookup_encode,
        )
        self.assertTrue(model.training)
        self.assertEqual(result["count"], 3)
        self.assertAlmostEqual(result["spearman"], 1.0, places=6)
        self.assertAlmostEqual(result["pearson"], 1.0, places=6)

    def test_metrics_logger_extends_legacy_schema(self):
        with tempfile.TemporaryDirectory() as run_dir:
            path = f"{run_dir}/metrics.csv"
            with open(path, "w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=["step", "event", "loss", "acc"])
                writer.writeheader()
                writer.writerow({"step": 1, "event": "val", "loss": 2.0, "acc": 0.5})

            logger = MetricsLogger(run_dir)
            logger.log(
                step=2,
                event="val",
                loss=1.0,
                acc=0.6,
                stsb_spearman=0.7,
                stsb_pearson=0.8,
            )
            logger.close()

            with open(path, newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertIn("stsb_spearman", rows[0])
            self.assertEqual(rows[0]["loss"], "2.0")
            self.assertEqual(rows[1]["stsb_spearman"], "0.7")
            self.assertEqual(rows[1]["stsb_pearson"], "0.8")


if __name__ == "__main__":
    unittest.main()
