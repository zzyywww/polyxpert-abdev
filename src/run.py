"""CLI interface for ESM2 Ridge baseline."""

from abdev_core import create_cli_app
from .model import PolyXpertModel


app = create_cli_app(PolyXpertModel, "PolyXpert")


if __name__ == "__main__":
    app()

