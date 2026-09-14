import unittest

from config import Config
from data_clip import _detect_pair_columns


class PairSourceConfigTests(unittest.TestCase):
    def test_new_pair_schemas_are_detected(self):
        for columns in (
            ("title1", "title2"),
            ("caption1", "caption2"),
            ("text", "simplified"),
        ):
            with self.subTest(columns=columns):
                self.assertEqual(_detect_pair_columns(columns), columns)

    def test_default_mix_contains_new_sources(self):
        self.assertEqual(
            Config().clip_dataset_specs[-3:],
            (
                ("sentence-transformers/stackexchange-duplicates", "title-title-pair"),
                ("sentence-transformers/coco-captions", "pair"),
                ("sentence-transformers/sentence-compression", "pair"),
            ),
        )


if __name__ == "__main__":
    unittest.main()
