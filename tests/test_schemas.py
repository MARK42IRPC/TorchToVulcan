from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


class JsonSchemaTests(unittest.TestCase):
    def test_graph_schema_is_valid_draft_2020_12(self) -> None:
        schema = load_json(REPOSITORY_ROOT / "schemas" / "graph-ir.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_model_package_schema_is_valid_draft_2020_12(self) -> None:
        schema = load_json(REPOSITORY_ROOT / "schemas" / "model-package.schema.json")
        Draft202012Validator.check_schema(schema)

    def test_relu_example_matches_graph_schema(self) -> None:
        schema = load_json(REPOSITORY_ROOT / "schemas" / "graph-ir.schema.json")
        example = load_json(REPOSITORY_ROOT / "examples" / "relu.graph.json")
        Draft202012Validator(schema).validate(example)

    def test_model_package_rejects_legacy_flat_manifest(self) -> None:
        schema = load_json(REPOSITORY_ROOT / "schemas" / "model-package.schema.json")
        legacy = {
            "format_version": "1.0",
            "graph_schema_version": "1.0",
            "model_name": "legacy",
        }

        self.assertFalse(Draft202012Validator(schema).is_valid(legacy))


if __name__ == "__main__":
    unittest.main()
