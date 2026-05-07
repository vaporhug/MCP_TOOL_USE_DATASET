This tools folder is organized into three layers.

- `astro_tools.py`
  Domain-specific workflow primitives for the Wolf 1069 discovery package.
- `data_processing.py`
  Generic table-reading, reshaping, and file-inventory helpers.
- `database_retrieval.py`
  Lightweight local/remote resource helpers for reference inspection and reproducible downloads.

Use the domain tool for the paper-specific analysis. Use the generic helper modules when the task needs standard table preparation or resource inspection before the main analysis.
