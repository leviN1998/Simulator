


class Logger:
    def __init__(self, path: str = "../data/logs/"):
        self.path = path
        self.error_file = open(f"{self.path}error.log", "a", buffering=1)
        self.info_file = open(f"{self.path}info.log", "a", buffering=1)
        self.debug_file = open(f"{self.path}debug.log", "a", buffering=1)
        self.progress_file = open(f"{self.path}progress.log", "a", buffering=1)

    def error(self, message: str):
        print(f"ERROR: {message}")
        self.error_file.write(f"ERROR: {message}\n")

    def info(self, message: str):
        print(f"INFO: {message}")
        self.info_file.write(f"INFO: {message}\n")
        # put into debug and progress as well
        self.debug_file.write(f"INFO: {message}\n")
        self.progress_file.write(f"INFO: {message}\n")

    def debug(self, message: str):
        print(f"DEBUG: {message}")
        self.debug_file.write(f"DEBUG: {message}\n")
        

    def progress(self, message: str):
        print(f"PROGRESS: {message}")
        self.progress_file.write(f"PROGRESS: {message}\n")
        # put into debug as well
        self.debug_file.write(f"PROGRESS: {message}\n")

    def close(self):
        self.error_file.close()
        self.info_file.close()
        self.debug_file.close()
        self.progress_file.close()
        print("Logger closed.")