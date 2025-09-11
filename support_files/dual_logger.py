import sys


class DualLogger:
    """Simple logger that writes both to the real stdout and to a UTF-8 log file.

    It preserves emojis in the file and writes console-safe replacements to the terminal
    when the terminal encoding doesn't support UTF-8.
    """

    def __init__(self, logfile_path: str):
        # Prefer the original system stdout to avoid nested wrappers
        self.terminal = getattr(sys, '__stdout__', sys.stdout)
        self.log_file = open(logfile_path, 'a', encoding='utf-8', errors='replace')

    def _console_safe(self, message: str) -> str:
        enc = getattr(self.terminal, 'encoding', '') or ''
        if enc.lower().startswith('utf'):
            return message
        # Fallback replacements for common emojis/symbols
        replacements = {
            '📊': '[CHART]', '💰': '[MONEY]', '📈': '[UP]', '📉': '[DOWN]',
            '🎯': '[TARGET]', '🚀': '[ROCKET]', '💎': '[DIAMOND]', '✅': '[OK]',
            '⚠️': '[WARNING]', '🎉': '[PARTY]', '💹': '[TRADING]', '💵': '[CASH]',
            '📝': '[NOTE]', '🖥️': '[SCREEN]', '🔥': '[HOT]', '⭐': '[STAR]',
            '₹': 'Rs.', '€': 'EUR', '$': 'USD', '£': 'GBP',
            '→': '->', '←': '<-', '↑': 'UP', '↓': 'DOWN',
            '•': '*', '…': '...', '–': '-', '—': '--'
        }
        out = message
        for k, v in replacements.items():
            out = out.replace(k, v)
        return out

    def write(self, message: str):
        try:
            self.terminal.write(self._console_safe(message))
        except Exception:
            try:
                sys.__stdout__.write(str(message))
            except Exception:
                pass

        try:
            m = message if (isinstance(message, str) and message.endswith('\n')) else f"{message}\n"
            self.log_file.write(m)
        except Exception:
            try:
                self.log_file.write(str(message) + '\n')
            except Exception:
                pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass
