# Manifest and runtime-config consistency.
#
# config/agent.yaml is the flat discovery manifest — every key at root level,
# `class:` a single dotted import path. config/config.yaml holds the runtime
# parameters the registry passes to the graph constructor. Both are live
# configuration, not documentation: the runtime file's retrieval block is
# forwarded into the inner graph, and the declared class has to BE the agent
# class in src/graph/graph.py. These tests pin that consistency so a drift
# fails here rather than at deploy time.
#
# Deterministic — no model call, no network.

import json
import pathlib

import yaml

from framework.schemas.trust_level import TrustLevel
from src.graph.graph import PlanServiceQAGraphNode, TelecomPlanServiceQAAgent, load_runtime_config
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_MANIFEST = yaml.safe_load((_ROOT / "config" / "agent.yaml").read_text(encoding="utf-8"))
_RUNTIME = yaml.safe_load((_ROOT / "config" / "config.yaml").read_text(encoding="utf-8"))


class TestManifestShape:
    def test_the_manifest_is_flat(self):
        """Nested keys are invisible to the registry — every key sits at root level."""
        assert "agent" not in _MANIFEST
        assert _MANIFEST["id"] == "TEL-C2-001"
        assert _MANIFEST["namespace"] == "tel"
        assert _MANIFEST["enabled"] is True

    def test_the_declared_class_is_the_agent_class(self):
        assert _MANIFEST["class"] == "src.graph.graph.TelecomPlanServiceQAAgent"
        module, _, class_name = _MANIFEST["class"].rpartition(".")
        assert module == TelecomPlanServiceQAAgent.__module__
        assert class_name == TelecomPlanServiceQAAgent.__name__
        assert _MANIFEST["name"] == TelecomPlanServiceQAAgent().name

    def test_category_industry_and_generation_mode(self):
        assert _MANIFEST["category"] == "Cat 2"
        assert _MANIFEST["industry"] == "TEL"
        assert _MANIFEST["generation_mode"] == "deterministic"

    def test_no_secret_or_extra_is_declared(self):
        """Answer assembly makes no model call and requires no provisioned secret."""
        assert _MANIFEST["requires"]["secrets"] == []
        assert _MANIFEST["requires"]["extras"] == []


class TestManifestSecurity:
    def test_the_declared_trust_level_matches_the_boundary_nodes(self):
        declared = TrustLevel(_MANIFEST["required_trust_level"])
        assert declared is TrustLevel.VERIFIED_EXTERNAL
        assert PreProcessNode.required_trust_level is declared
        assert PostProcessNode.required_trust_level is declared


class TestRuntimeConfig:
    def test_runtime_parameters_live_in_the_runtime_file(self):
        assert "max_retry" not in _MANIFEST
        assert _RUNTIME["max_retry"] == 3
        assert _RUNTIME["timeout_s"] == 30

    def test_max_retry_is_within_the_framework_ceiling(self):
        max_retry = _RUNTIME["max_retry"]
        assert isinstance(max_retry, int)
        assert 0 <= max_retry < 10

    def test_human_in_the_loop_is_not_enabled(self):
        assert (_RUNTIME.get("hitl") or {}).get("enabled", False) is False

    def test_load_runtime_config_reads_the_file(self):
        loaded = load_runtime_config()
        assert loaded["max_retry"] == _RUNTIME["max_retry"]
        assert loaded["retrieval"] == _RUNTIME["retrieval"]


class TestRetrievalBlock:
    def test_the_retrieval_block_matches_the_node_defaults(self):
        """A drift between the file and the module defaults silently changes tuning."""
        from src.nodes.rerank_filter_node import _DEFAULT_SCORE_THRESHOLD, _DEFAULT_TOP_K
        from src.nodes.retrieve_node import _DEFAULT_RETRIEVAL as retrieve_defaults

        retrieval = _RUNTIME["retrieval"]
        assert retrieval["top_k"] == retrieve_defaults["top_k"] == _DEFAULT_TOP_K
        assert retrieval["score_threshold"] == retrieve_defaults["score_threshold"] == _DEFAULT_SCORE_THRESHOLD
        assert retrieval["kb_path"] == retrieve_defaults["kb_path"]
        assert (_ROOT / retrieval["kb_path"]).is_file()

    def test_parent_config_forwards_the_runtime_blocks(self):
        forwarded = PlanServiceQAGraphNode()._parent_config()["configurable"]
        assert forwarded["retrieval"] == _RUNTIME["retrieval"]
        assert forwarded["llm"] == _RUNTIME["llm"]
        assert forwarded["retrieval"], "the forwarded retrieval block must never be empty"

    def test_parent_config_falls_back_when_the_file_is_unreadable(self, monkeypatch):
        monkeypatch.setattr("src.graph.graph.load_runtime_config", lambda: {})
        forwarded = PlanServiceQAGraphNode()._parent_config()["configurable"]
        assert forwarded["retrieval"]["top_k"] == _RUNTIME["retrieval"]["top_k"]
        assert forwarded["llm"]["max_tokens"] == _RUNTIME["llm"]["max_tokens"]


class TestBundledCatalog:
    def test_the_bundled_catalog_is_a_well_formed_entry_list(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        assert isinstance(entries, list)
        assert len(entries) >= 5, "the bundled catalog must carry a usable corpus"
        for entry in entries:
            assert set(entry.keys()) == {"id", "title", "category", "source", "tags", "content"}
            assert entry["id"] and entry["title"] and entry["content"]

    def test_catalog_identifiers_are_unique(self):
        entries = json.loads((_ROOT / _RUNTIME["retrieval"]["kb_path"]).read_text(encoding="utf-8"))
        ids = [e["id"] for e in entries]
        assert len(ids) == len(set(ids))
