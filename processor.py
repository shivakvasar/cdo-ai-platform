import os
import time

UPLOAD_DIR = "/app/uploads"
POLL_INTERVAL = 5


def main():
    print(f"processor started, watching {UPLOAD_DIR}", flush=True)
    seen = set(os.listdir(UPLOAD_DIR)) if os.path.isdir(UPLOAD_DIR) else set()
    while True:
        if os.path.isdir(UPLOAD_DIR):
            current = set(os.listdir(UPLOAD_DIR))
            for new_file in current - seen:
                print(f"new file detected: {new_file}", flush=True)
            seen = current
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
