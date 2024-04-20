from .artifacts import LogArtifactsCallback
from .checkpoints import load_checkpoint_callbacks
from .test_report import GenerateTestReportCallback
from .validations import ValidateArchitectureCallback

key_to_callback_class = {
    "classification-generate_test_report": GenerateTestReportCallback,
    "classification-validate_architecture": ValidateArchitectureCallback,
    "common-log_artifacts_callback": LogArtifactsCallback,
}
