import os
import tempfile

from src.utils.logging import get_logger, set_logging_config


def main():
    set_logging_config(level="INFO", debug=False, timestamp=False, color=False)
    logger = get_logger("CentralizedLoggerTest")

    logger.info("Info message", phase="test")
    logger.warn("Warning message", retry=1)
    logger.error("Error message", reason="simulated")
    logger.success("Success message", status="ok")
    logger.metric("Metric message", ap70=0.87, comm_ratio=0.0953)
    logger.config("Config message", preset="phase5_receiver_request_topk_10")
    logger.run("Run message", step="unit_test")
    logger.command("Command message", cmd="python -m src.tools.inference")
    logger.save("Save message", path="/tmp/fake")

    # debug disabled here: should not print
    logger.debug("Debug hidden", key="value")

    # enable debug and validate no crash
    set_logging_config(level="DEBUG", debug=True, timestamp=False, color=False)
    dbg = get_logger("CentralizedLoggerDebug")
    dbg.debug("Debug visible", value=42)

    # file logging check
    with tempfile.TemporaryDirectory() as tmp:
        log_file = os.path.join(tmp, "logger_test.log")
        file_logger = get_logger(
            "CentralizedLoggerFile",
            level="INFO",
            debug=False,
            timestamp=False,
            color=False,
            log_to_file=True,
            file_path=log_file,
        )
        file_logger.info("File logging works", output=log_file)
        assert os.path.exists(log_file), "Log file not created"
        text = open(log_file, "r").read()
        assert "[INFO] [CentralizedLoggerFile]" in text

    print("Centralized logger test passed")


if __name__ == "__main__":
    main()
