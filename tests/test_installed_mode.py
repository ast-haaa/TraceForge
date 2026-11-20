"""Tests for installed-mode template resolution across working directories."""
from pathlib import Path

from agentreplay import api


def test_package_templates_directory_exists_and_complete():
    """Verify templates directory is packaged with the library and has all required templates."""
    template_dir = Path(api._TEMPLATE_DIR)
    assert template_dir.exists(), f"Template directory does not exist: {template_dir}"
    assert template_dir.is_dir(), f"Template path is not a directory: {template_dir}"

    required_templates = [
        "index.html",
        "trace.html",
        "fork_diff.html",
        "test_view.html",
    ]
    for tmpl in required_templates:
        tmpl_file = template_dir / tmpl
        assert tmpl_file.exists(), f"Missing template: {tmpl}"
        assert tmpl_file.stat().st_size > 0, f"Template {tmpl} is empty"


def test_templates_render_when_cwd_is_different(tmp_path, monkeypatch):
    """Verify Jinja2 templates render cleanly when the current working directory changes."""
    # Change working directory to a separate temporary directory
    monkeypatch.chdir(tmp_path)
    assert Path.cwd() == tmp_path

    # Verify Jinja template resolution still succeeds from memory / package path
    tmpl = api.templates.get_template("index.html")
    rendered = tmpl.render(traces=[], read_only=True)
    assert "AgentReplay" in rendered
    assert "Read-only demo mode" in rendered
