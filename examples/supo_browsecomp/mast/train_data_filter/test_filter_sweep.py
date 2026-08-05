import argparse
import unittest

from examples.supo_browsecomp.mast.train_data_filter.filter_sweep import (
    DEV_STAGE,
    build_mast_command,
    frozen_config,
)


class FilterSweepConfigTest(unittest.TestCase):
    def test_base_only_fixed_tool_protocol_is_frozen_and_forwarded(self) -> None:
        args = argparse.Namespace(
            base_only=True,
            checkpoint_only=False,
            run_name="supo_4b_n8_8n_40iter_dump_groupfix-mj1d0qw1",
            steps=[4, 9],
            fixed_search_topk=5,
            doc_words_full=10000,
            tenant="rhea_assistant_avocado_iterations",
            tenant_path=None,
            priority="HIGH",
            search_addr_file="/mnt/wsfuse/hhzhang01/supo-slime/search-server.addr",
            search_server_tenant="rhea_assistant_interns",
            compare_query_ids=None,
            comparison_name="reference",
        )
        config = frozen_config("test-batch", args)

        self.assertEqual(config["points"], [{"name": "base", "step": "base"}])
        self.assertEqual(config["evaluation"]["fixed_search_topk"], 5)
        self.assertEqual(config["evaluation"]["doc_words_full"], 10000)
        self.assertEqual(config["mast"]["tenant"], "rhea_assistant_avocado_iterations")
        self.assertEqual(config["mast"]["priority"], "HIGH")

        command = build_mast_command(
            config,
            DEV_STAGE / "train-data-filter/test-batch",
            DEV_STAGE / "train-data-filter-code/test.tgz",
            "digest",
            config["points"][0],
            dry_run=True,
        )
        custom_command = next(arg for arg in command if arg.startswith("--docker_custom_cmd="))
        self.assertIn("BCPLUS_FIXED_SEARCH_TOPK=5", custom_command)
        self.assertIn("BCPLUS_DOC_WORDS_FULL=10000", custom_command)
        self.assertIn("FILTER_N=8", custom_command)
        self.assertIn("--tenant=rhea_assistant_avocado_iterations", command)
        self.assertIn("--job_priority=HIGH", command)
        self.assertEqual(command[-1], "--dryrun")


if __name__ == "__main__":
    unittest.main()
