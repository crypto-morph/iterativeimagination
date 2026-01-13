from pathlib import Path

from flask import Flask


def create_app() -> Flask:
    base_path = Path(__file__).resolve().parent.parent
    app = Flask(
        __name__,
        template_folder=str(base_path / "templates"),
        static_folder=str(base_path / "static"),
    )

    from . import legacy_routes  # noqa: WPS433

    legacy_routes.register(app)

    return app
