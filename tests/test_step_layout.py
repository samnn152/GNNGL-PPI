from __future__ import annotations

import ast
import unittest
from pathlib import Path


class StepLayoutTest(unittest.TestCase):
    """Protect the one-step-per-module training pipeline layout."""

    def test_each_step_module_starts_with_exactly_one_pipeline_step(self) -> None:
        """Ensure every step module exports exactly one leading pipeline step class."""
        steps_dir = Path(__file__).parents[1] / 'src' / 'train' / 'models' / 'steps'
        for path in sorted(steps_dir.rglob('*.py')):
            if path.name == '__init__.py':
                continue
            with self.subTest(path=path.relative_to(steps_dir)):
                module = ast.parse(path.read_text(encoding='utf-8'))
                classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
                self.assertTrue(classes, 'step module contains no class')
                first_class = classes[0]
                pipeline_steps = [
                    node
                    for node in classes
                    if any(isinstance(base, ast.Name) and base.id == 'PipelineStep' for base in node.bases)
                ]
                self.assertEqual(len(pipeline_steps), 1, 'step module must contain exactly one PipelineStep')
                self.assertIs(first_class, pipeline_steps[0], 'PipelineStep must be the first class in the module')


if __name__ == '__main__':
    unittest.main()
