from __future__ import annotations

import json
import unittest
from pathlib import Path

from torch_to_vulcan.ir import GraphValidationError, graph_from_dict


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class GraphIrTests(unittest.TestCase):
    def load_example(self) -> dict[str, object]:
        with (REPOSITORY_ROOT / "examples" / "relu.graph.json").open(encoding="utf-8") as stream:
            return json.load(stream)

    def test_example_is_valid_and_round_trips(self) -> None:
        value = self.load_example()
        graph = graph_from_dict(value)

        self.assertEqual(graph.model.name, "relu_example")
        self.assertEqual(graph.to_dict(), value)

    def test_rejects_unknown_input_tensor(self) -> None:
        value = self.load_example()
        value["nodes"][0]["inputs"] = ["missing"]  # type: ignore[index]

        with self.assertRaisesRegex(GraphValidationError, "unknown tensor 'missing'"):
            graph_from_dict(value)

    def test_rejects_non_topological_node_order(self) -> None:
        value = self.load_example()
        value["tensors"]["intermediate"] = {  # type: ignore[index]
            "dtype": "float32",
            "shape": [1],
            "layout": "C",
        }
        value["nodes"][0]["inputs"] = ["intermediate"]  # type: ignore[index]

        with self.assertRaisesRegex(GraphValidationError, "topologically ordered"):
            graph_from_dict(value)


if __name__ == "__main__":
    unittest.main()

