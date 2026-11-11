import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import reproduce


class WorkflowTests(unittest.TestCase):
    def test_external_training_inputs_reach_each_stage_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "label cache"
            labels.mkdir()
            (labels / "manifest.json").write_text("{}", encoding="utf-8")
            args = SimpleNamespace(
                action="train",
                data_root=root / "dataset parent",
                part_labels=labels,
                pretrained=root / "weights.pth",
                source_supervision=root / "source labels",
                work_dir=root / "run",
            )
            data_root = args.data_root / "Occluded-Duke"
            with (
                patch.object(reproduce, "check_pretrained", return_value="pretrain"),
                patch.object(reproduce, "load_supervision", return_value=(None, None, {"weak": "points"})),
                patch.object(reproduce, "dataset_manifest", return_value=(data_root, {})),
                patch.object(reproduce, "command") as command,
            ):
                reproduce.train(args)
                calls = [call.args for call in command.call_args_list]
                self.assertEqual([c[2] for c in calls[:4]], ["check-labels", "p120", "r180", "calibrate"])
                self.assertTrue(all(data_root in c for c in calls[:4]))
                self.assertEqual(calls[4][0], "portable_model.py")
                self.assertEqual(calls[5][0], "train_evidence.py")
                self.assertIn(args.source_supervision, calls[5])

                command.reset_mock()
                args.action = "resume"
                reproduce.train(args)
                self.assertEqual(len(command.call_args_list), 5)
                self.assertIn("--resume", command.call_args.args)

    def test_evaluation_requires_no_training_annotations(self):
        with patch("sys.argv", ["reproduce.py", "test", "--data-root", "duke", "--output", "eval"]):
            args = reproduce.parse_arguments()
        self.assertIsNone(args.reference)
        self.assertEqual(args.action, "test")


if __name__ == "__main__":
    unittest.main()
