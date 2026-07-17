from .aggregate import aggregate_run, write_tables  # noqa: F401
from .charts import generate_charts  # noqa: F401
from .export import export_run  # noqa: F401
from .report_md import generate_summary_md  # noqa: F401


def generate_full_report(run_dir) -> None:
    """Aggregate saved metrics and produce tables, charts, and summary.md.
    Never loads ASR or diarization models."""
    tables = aggregate_run(run_dir)
    write_tables(run_dir, tables)
    generate_charts(run_dir, tables)
    generate_summary_md(run_dir, tables)
