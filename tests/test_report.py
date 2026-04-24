"""
test_report.py – Tests for CSV report generation and analysis.
"""

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.report import generate_csv, analyze_results


@pytest.fixture
def sample_results():
    """Sample compression results for testing."""
    return [
        {
            "Filename": "test.png",
            "Format": "JPEG",
            "Quality_CRF": 30,
            "Original_Size_Bytes": 100000,
            "Compressed_Size_Bytes": 20000,
            "Compression_Ratio": 5.0,
            "PSNR": 32.5,
            "SSIM": 0.92,
        },
        {
            "Filename": "test.png",
            "Format": "JPEG",
            "Quality_CRF": 70,
            "Original_Size_Bytes": 100000,
            "Compressed_Size_Bytes": 40000,
            "Compression_Ratio": 2.5,
            "PSNR": 38.0,
            "SSIM": 0.97,
        },
        {
            "Filename": "test.png",
            "Format": "WEBP",
            "Quality_CRF": 30,
            "Original_Size_Bytes": 100000,
            "Compressed_Size_Bytes": 15000,
            "Compression_Ratio": 6.67,
            "PSNR": 33.0,
            "SSIM": 0.93,
        },
        {
            "Filename": "test.png",
            "Format": "WEBP",
            "Quality_CRF": 70,
            "Original_Size_Bytes": 100000,
            "Compressed_Size_Bytes": 30000,
            "Compression_Ratio": 3.33,
            "PSNR": 39.0,
            "SSIM": 0.98,
        },
    ]


class TestGenerateCSV:
    """Tests for CSV report generation."""

    def test_creates_csv_file(self, sample_results, output_dir):
        path = os.path.join(output_dir, "test_report.csv")
        result = generate_csv(sample_results, path)
        assert os.path.isfile(result)

    def test_csv_has_correct_rows(self, sample_results, output_dir):
        import pandas as pd
        path = os.path.join(output_dir, "test_rows.csv")
        generate_csv(sample_results, path)
        df = pd.read_csv(path)
        assert len(df) == 4

    def test_csv_has_required_columns(self, sample_results, output_dir):
        import pandas as pd
        path = os.path.join(output_dir, "test_cols.csv")
        generate_csv(sample_results, path)
        df = pd.read_csv(path)
        required = {"Filename", "Format", "Quality_CRF", "PSNR", "SSIM", "Compression_Ratio"}
        assert required.issubset(set(df.columns))

    def test_empty_results(self, output_dir):
        path = os.path.join(output_dir, "empty.csv")
        result = generate_csv([], path)
        assert result == path

    def test_creates_parent_directory(self, test_dir):
        path = os.path.join(test_dir, "report_subdir", "nested", "report.csv")
        generate_csv([{
            "Filename": "x.png", "Format": "JPEG", "Quality_CRF": 50,
            "Original_Size_Bytes": 100, "Compressed_Size_Bytes": 50,
            "Compression_Ratio": 2.0, "PSNR": 30.0, "SSIM": 0.9,
        }], path)
        assert os.path.isfile(path)


class TestAnalyzeResults:
    """Tests for the analysis report generator."""

    def test_analysis_returns_string(self, sample_results, output_dir):
        path = os.path.join(output_dir, "analysis_src.csv")
        generate_csv(sample_results, path)
        analysis = analyze_results(path)
        assert isinstance(analysis, str)
        assert len(analysis) > 100

    def test_analysis_contains_format_names(self, sample_results, output_dir):
        path = os.path.join(output_dir, "analysis_fmts.csv")
        generate_csv(sample_results, path)
        analysis = analyze_results(path)
        assert "JPEG" in analysis
        assert "WEBP" in analysis or "WebP" in analysis

    def test_analysis_contains_recommendations(self, sample_results, output_dir):
        path = os.path.join(output_dir, "analysis_recs.csv")
        generate_csv(sample_results, path)
        analysis = analyze_results(path)
        assert "RECOMMENDATIONS" in analysis
