"""Temporary compatibility layer for the viewer refactor."""

from __future__ import annotations

from flask import Flask


def register(app: Flask) -> None:
    """Register routes on the Flask app."""

    from .routes.pages import pages_bp
    from .api.project_api import project_api_bp
    from .api.aivis_api import aivis_api_bp
    from .api.mask_api import mask_api_bp
    from .api.aigen_api import aigen_api_bp
    from .api.prompt_suggestions_api import prompt_suggestions_api_bp
    from .api.live_api import live_api_bp
    from .api.iteration_api import iteration_api_bp
    from .api.run_api import run_api_bp
    from .api.config_api import config_api_bp

    app.register_blueprint(pages_bp)
    app.register_blueprint(project_api_bp)
    app.register_blueprint(aivis_api_bp)
    app.register_blueprint(mask_api_bp)
    app.register_blueprint(aigen_api_bp)
    app.register_blueprint(prompt_suggestions_api_bp)
    app.register_blueprint(live_api_bp)
    app.register_blueprint(iteration_api_bp)
    app.register_blueprint(run_api_bp)
    app.register_blueprint(config_api_bp)
