import logging
import threading

import pytest

from jasna.gui.app import GUILogHandler, JasnaApp
from jasna.gui.models import JobStatus
from jasna.gui.processor import ProgressUpdate
from jasna.gui.thread_dispatch import GuiThreadDispatcher


def test_background_post_does_not_execute_until_gui_thread_drains():
    dispatcher = GuiThreadDispatcher()
    calls = []

    worker = threading.Thread(
        target=lambda: dispatcher.post(calls.append, (threading.get_ident(), "worker"))
    )
    worker.start()
    worker.join()

    assert calls == []
    queued = dispatcher.take()
    assert len(queued) == 1
    queued[0].callback(*queued[0].args, **queued[0].kwargs)
    assert calls[0][1] == "worker"


def test_dispatcher_rejects_callbacks_and_discards_pending_work_after_close():
    dispatcher = GuiThreadDispatcher()
    dispatcher.post(lambda: None)

    dispatcher.close()

    assert dispatcher.take() == ()
    assert dispatcher.post(lambda: None) is False


def test_dispatcher_can_only_be_drained_by_its_owner_thread():
    dispatcher = GuiThreadDispatcher()
    errors = []

    def drain_from_worker():
        try:
            dispatcher.take()
        except Exception as error:
            errors.append(error)

    worker = threading.Thread(target=drain_from_worker)
    worker.start()
    worker.join()

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)


def test_gui_log_handler_posts_without_calling_tk_from_worker():
    posted = []

    class LogPanel:
        def add_log(self, level, message):
            raise AssertionError("callback must stay queued in this test")

        def after_idle(self, *_args, **_kwargs):
            raise AssertionError("worker thread entered Tk")

    panel = LogPanel()
    handler = GUILogHandler(
        panel,
        lambda callback, *args: posted.append((callback, args)),
    )
    record = logging.LogRecord(
        "jasna.test",
        logging.INFO,
        __file__,
        1,
        "worker log",
        (),
        None,
    )

    worker = threading.Thread(target=handler.emit, args=(record,))
    worker.start()
    worker.join()

    assert posted == [(panel.add_log, ("INFO", "worker log"))]


@pytest.mark.parametrize(
    ("method_name", "target_name", "args"),
    [
        (
            "_on_processor_progress",
            "_handle_progress",
            (ProgressUpdate(job_id=7, status=JobStatus.PROCESSING),),
        ),
        ("_on_processor_log", "_add_log", ("INFO", "message")),
        ("_on_processor_complete", "_handle_complete", ()),
    ],
)
def test_processor_callbacks_only_post_to_gui(method_name, target_name, args):
    posted = []
    run_logs = []

    class App:
        def _post_to_gui(self, callback, *callback_args):
            posted.append((callback, callback_args))

        def _handle_progress(self, update):
            raise AssertionError("callback must stay queued in this test")

        def _add_log(self, level, message):
            raise AssertionError("callback must stay queued in this test")

        def _handle_complete(self):
            raise AssertionError("callback must stay queued in this test")

        def _enqueue_run_log(self, level, message):
            run_logs.append((level, message))

    app = App()
    app._log_panel = type("Panel", (), {"add_log": app._add_log})()

    getattr(JasnaApp, method_name)(app, *args)

    target = getattr(app, target_name)
    if method_name == "_on_processor_log":
        target = app._log_panel.add_log
    assert posted == [(target, args)]
    assert run_logs == ([args] if method_name == "_on_processor_log" else [])
