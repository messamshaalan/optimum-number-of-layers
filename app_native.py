"""LayerOptimum Pro — native desktop window entry point (for the .exe build)."""

import multiprocessing

import app_nicegui  # noqa: F401  (builds the UI as an import-time side effect)
from nicegui import native, ui

if __name__ in {'__main__', '__mp_main__'}:
    multiprocessing.freeze_support()
    ui.run(
        title='LayerOptimum Pro',
        dark=True,
        reload=False,
        native=True,
        window_size=(1500, 950),
        port=native.find_open_port(),
        favicon='⛽',
    )
