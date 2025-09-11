import sys
import io

class DualLogger:
    """
    Writes to both terminal (stdout) and a log file using UTF-8.
    Opens the log file with encoding='utf-8' and errors='replace' so emojis/unicode
    are preserved and unknown glyphs are replaced safely.
    """
    def __init__(self, logfile_path):
        # keep a reference to the original stdout for terminal writes
        self.terminal = sys.stdout
        # open file in append mode with utf-8 and replace errors
        self.log_file = open(logfile_path, "a", encoding="utf-8", errors="replace")

    def write(self, message):
        # write to terminal as-is (best-effort)
        try:
            self.terminal.write(message)
        except Exception:
            try:
                # fallback to the original system stdout
                sys.__stdout__.write(message)
            except Exception:
                pass

        # write to log file (utf-8, with replacement already set on open)
        try:
            self.log_file.write(message)
        except Exception:
            # defensive fallback: ensure we write a safe string
            try:
                safe = message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                self.log_file.write(safe)
            except Exception:
                # last resort: ignore the write error
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




