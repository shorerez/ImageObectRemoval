"""Application bootstrap."""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from .config import APP_NAME
from .ui.main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    argv = list(sys.argv if argv is None else argv)
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName(APP_NAME)

    window = MainWindow()
    window.show()

    args = [a for a in argv[1:] if not a.startswith("-")]
    if args:
        window.open_path(args[0])

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
