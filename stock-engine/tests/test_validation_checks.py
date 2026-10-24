"""Tests for validation checks: check_outputs.py and gen_results_md.py."""
import json
import tempfile
import pytest
from pathlib import Path

from pipeline.check_outputs import check_outputs
from pipeline.gen_results_md import main as gen_results_main


def test_check_outputs_passes_on_clean_dir():
    """check_outputs should pass on a directory with valid JSON files and no banned tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        
        # Create required files with clean content
        required = [
            "five_baselines_benchmark.json",
            "factor_backtest_metrics.json",
            "h2_predictive.json",
            "signal_decay_analysis.json",
            "propagation_cascade_analysis.json",
            "robustness_tests.json",
            "inter_annotator_agreement.json",
        ]
        for fname in required:
            (tables_dir / fname).write_text(json.dumps({"metric": 0.01, "p": 0.5}))
        
        # Should pass
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert check_outputs() == 0
        finally:
            os.chdir(old_cwd)


def test_check_outputs_fails_on_missing_dir():
    """check_outputs should fail if outputs/tables doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert check_outputs() == 1
        finally:
            os.chdir(old_cwd)


def test_check_outputs_fails_on_empty_dir():
    """check_outputs should fail if outputs/tables is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert check_outputs() == 1
        finally:
            os.chdir(old_cwd)


def test_check_outputs_fails_on_missing_required():
    """check_outputs should fail if required files are missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        (tables_dir / "five_baselines_benchmark.json").write_text(json.dumps({"metric": 0.01}))
        
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert check_outputs() == 1
        finally:
            os.chdir(old_cwd)


def test_check_outputs_fails_on_banned_token():
    """check_outputs should fail if a file contains banned tokens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        
        required = [
            "five_baselines_benchmark.json",
            "factor_backtest_metrics.json",
            "h2_predictive.json",
            "signal_decay_analysis.json",
            "propagation_cascade_analysis.json",
            "robustness_tests.json",
            "inter_annotator_agreement.json",
        ]
        for fname in required:
            (tables_dir / fname).write_text(json.dumps({"metric": 0.01}))
        
        # Add a file with banned token
        (tables_dir / "bad_file.json").write_text(json.dumps({"_placeholder": True}))
        
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert check_outputs() == 1
        finally:
            os.chdir(old_cwd)


def test_check_outputs_fails_on_simulated_token():
    """check_outputs should fail on 'simulated' token."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        
        required = [
            "five_baselines_benchmark.json",
            "factor_backtest_metrics.json",
            "h2_predictive.json",
            "signal_decay_analysis.json",
            "propagation_cascade_analysis.json",
            "robustness_tests.json",
            "inter_annotator_agreement.json",
        ]
        for fname in required:
            (tables_dir / fname).write_text(json.dumps({"metric": 0.01}))
        
        # Add a file with 'simulated' in content
        (tables_dir / "bad_file.json").write_text(json.dumps({"note": "this is simulated data"}))
        
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert check_outputs() == 1
        finally:
            os.chdir(old_cwd)


def test_gen_results_md_fails_on_missing_dir():
    """gen_results_md should fail if outputs/tables doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert gen_results_main() == 1
        finally:
            os.chdir(old_cwd)


def test_gen_results_md_fails_on_empty_dir():
    """gen_results_md should fail if outputs/tables is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert gen_results_main() == 1
        finally:
            os.chdir(old_cwd)


def test_gen_results_md_succeeds_on_valid_dir():
    """gen_results_md should succeed on valid input."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tables_dir = Path(tmpdir) / "outputs" / "tables"
        tables_dir.mkdir(parents=True)
        (tables_dir / "test.json").write_text(json.dumps({"metric": 0.01}))
        
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmpdir)
            assert gen_results_main() == 0
            # Check output file was created
            out_file = Path(tmpdir) / "RESULTS.md"
            assert out_file.exists()
            content = out_file.read_text()
            assert "test.json" in content
            assert "metric" in content
        finally:
            os.chdir(old_cwd)