import unittest
import tempfile
from pathlib import Path
try:
    from release_contract import validate_lineage, PIPELINE
except ImportError:
    validate_lineage = None
    PIPELINE = []
try:
    from release_contract import check_output_directory, reconcile_endpoint
except ImportError:
    check_output_directory = reconcile_endpoint = None


class ReleaseContractTests(unittest.TestCase):
    def test_new_checks_cannot_overwrite_immutable_packaged_verification(self):
        self.assertIsNotNone(check_output_directory)
        root=Path(tempfile.gettempdir())/'reid_contract_fixture'
        self.assertTrue(check_output_directory(root).is_relative_to(root/'runs'))
        self.assertFalse(check_output_directory(root).is_relative_to(root/'verification'))

    def test_completed_resume_recovers_missing_endpoint_without_overwriting_conflicts(self):
        self.assertIsNotNone(reconcile_endpoint)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);(root/'last.pt').write_bytes(b'committed-complete-resume')
            endpoint=reconcile_endpoint(root,6)
            self.assertEqual(endpoint.read_bytes(),b'committed-complete-resume')
            self.assertEqual(reconcile_endpoint(root,6),endpoint)
            endpoint.unlink()  # Simulate an independent conflicting file, not an in-place write through the hard link.
            endpoint.write_bytes(b'different-immutable-checkpoint')
            with self.assertRaises(ValueError):reconcile_endpoint(root,6)
            self.assertEqual(endpoint.read_bytes(),b'different-immutable-checkpoint')
            self.assertEqual((root/'last.pt').read_bytes(),b'committed-complete-resume')

    def test_complete_pipeline_includes_refinement_calibration_and_trained_evidence(self):
        self.assertEqual(PIPELINE, ['p120', 'r180', 'train_bn', 'frozen_evidence', 'train_quality', 'full_gallery_graph'])

    def test_old_native_only_or_different_parent_is_rejected(self):
        self.assertIsNotNone(validate_lineage)
        native = dict(format='complete_evidence_native_v1', model_sha256='a'*64,
                      public_pretraining_sha256='b'*64, p120_epoch=120, r4_epoch=180,
                      train_only_bn_images=15618, context_residual=False)
        head = dict(format='target_allocation_resume_v1', epoch=6,
                    frozen_parent=dict(model_sha256='a'*64),
                    history=[dict(nonzero_gradient_steps=20, parameter_delta=.1)])
        validate_lineage(native, head)
        for change in [dict(r4_epoch=0), dict(train_only_bn_images=0), dict(context_residual=True)]:
            with self.assertRaises(ValueError): validate_lineage(dict(native, **change), head)
        bad = dict(head, frozen_parent=dict(model_sha256='c'*64))
        with self.assertRaises(ValueError): validate_lineage(native, bad)


if __name__ == '__main__': unittest.main()
