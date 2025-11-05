import logging
import sys
import types

# ----------------------------------------------------------------------------
# CustomFormatter class
# ----------------------------------------------------------------------------
class CustomFormatter(logging.Formatter):
    """A custom log formatter with ANSI colors."""
    
    grey = "\x1b[38m"
    pink = "\x1b[95m"
    turquoise = "\x1b[96m"
    green = "\x1b[92m"
    yellow = "\x1b[93m"
    red = "\x1b[91m"
    bold_red = "\x1b[91;1m"
    reset = "\x1b[0m"
    
    # Updated format_str to be cleaner
    format_str = "[%(levelname)s] - %(message)s"

    FORMATS = {
        logging.DEBUG: pink + format_str + reset,
        logging.INFO: green + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        """Applies the correct color format based on log level."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# ----------------------------------------------------------------------------
# Newline logic
# ----------------------------------------------------------------------------
def log_newline(self, num_lines=1):
    """
    Swaps handlers to log blank lines, then swaps back.
    This method is bound to the logger instance by create_logger.
    """
    if hasattr(self, 'console_handler') and hasattr(self, 'blank_handler'):
        # Switch handler, output blank lines
        self.removeHandler(self.console_handler)
        self.addHandler(self.blank_handler)
        
        for _ in range(num_lines):
            self.info('') # Log an empty string

        # Switch back to the standard console handler
        self.removeHandler(self.blank_handler)
        self.addHandler(self.console_handler)
    else:
        # Fallback in case handlers aren't set up correctly
        for _ in range(num_lines):
            self.info('')

# ----------------------------------------------------------------------------
# Logger creation function
# ----------------------------------------------------------------------------
def create_logger(name="My_app", log_level=logging.DEBUG):
    """
    Creates a logger with colored output and a .newline() method.
    
    Args:
        name (str): The name for the logger.
        log_level (int): The logging level (e.g., logging.DEBUG).
    
    Returns:
        logging.Logger: A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    logger.propagate = False # Prevents duplicate logs in parent loggers

    # Only add handlers if the logger doesn't already have them
    if not logger.handlers:
        
        # 1. The main handler with custom colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(CustomFormatter())
        
        # 2. The handler for logging blank lines (with no format)
        blank_handler = logging.StreamHandler(sys.stdout)
        blank_handler.setLevel(log_level)
        blank_handler.setFormatter(logging.Formatter(fmt=''))

        # Add the main handler by default
        logger.addHandler(console_handler)

        # --------------------------------------------------------------------
        # Store the handlers on the logger object itself so log_newline
        # can access and swap them.
        # --------------------------------------------------------------------
        logger.console_handler = console_handler
        logger.blank_handler = blank_handler
        
        # Bind the log_newline function as a method on this logger instance
        logger.newline = types.MethodType(log_newline, logger)

    return logger
