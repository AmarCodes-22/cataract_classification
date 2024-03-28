from .checkpoints import load_checkpoint_callbacks
from .test_report import GenerateTestReportCallback

key_to_callback_class = {"classification-generate_test_report": GenerateTestReportCallback}
