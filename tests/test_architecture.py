from __future__ import annotations

import ast
import unittest
from pathlib import Path


def imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]


class FeatureFirstMVCTest(unittest.TestCase):
    """Protect feature ownership and MVC dependency boundaries."""

    def test_top_level_source_packages_are_feature_first(self) -> None:
        src = Path(__file__).parents[1] / 'src'
        packages = {
            path.name
            for path in src.iterdir()
            if path.is_dir() and (path / '__init__.py').exists()
        }
        self.assertEqual(packages, {'common', 'train', 'test'})

    def test_feature_models_do_not_import_controllers_or_views(self) -> None:
        root = Path(__file__).parents[1]
        cases = {
            root / 'src' / 'train' / 'models': ('src.train.controllers', 'src.train.views'),
            root / 'src' / 'test' / 'models': ('src.test.controllers', 'src.test.views'),
            root / 'src' / 'common' / 'models': (
                'src.train.controllers', 'src.train.views',
                'src.test.controllers', 'src.test.views',
            ),
        }
        for models, forbidden in cases.items():
            for path in sorted(models.rglob('*.py')):
                with self.subTest(path=path.relative_to(root)):
                    violations = [name for name in imported_modules(path) if name.startswith(forbidden)]
                    self.assertEqual(violations, [])

    def test_views_do_not_control_training(self) -> None:
        root = Path(__file__).parents[1]
        views = root / 'src' / 'train' / 'views'
        for path in sorted(views.rglob('*.py')):
            with self.subTest(path=path.relative_to(root)):
                imports = imported_modules(path)
                self.assertFalse(any(name.startswith('src.train.controllers') for name in imports))


if __name__ == '__main__':
    unittest.main()
