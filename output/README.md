# Output Directory

This directory contains generated reports and results from the fact-checking system.

## Structure

```
output/
└── reports/      # Fact-check reports in JSON and Markdown formats
```

## Reports

### Generated Files

The Reporter Agent (Agent 5) automatically generates reports here:

- **JSON format**: `report_{claim_id}_{timestamp}.json`
- **Markdown format**: `report_{claim_id}_{timestamp}.md`

### Report Contents

Each report includes:
- Original claim
- Verdict (SUPPORTED/REFUTED/INSUFFICIENT_INFO/CONFLICTING)
- Confidence score
- Evidence summary
- Key findings
- Recommended actions
- Complete audit trail

### Example Usage

```python
from agents.reporter import ReporterAgent

reporter = ReporterAgent()
result = reporter.generate_report(fact_check_result)

# Automatically saved to output/reports/
```

### Retention Policy

- Reports are kept indefinitely by default
- Configure retention in `config/settings.py`
- Use `scripts/cleanup_reports.py` to manage old reports

## .gitignore

This directory is git-ignored to avoid committing generated reports to the repository.

## Export Formats

Supported export formats:
- JSON (default)
- Markdown
- PDF (requires additional dependencies)
- HTML (via API)
