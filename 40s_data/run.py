import subprocess
import sys
import os

subprocess.run(["clear_processed_data.bat"])
subprocess.run([sys.executable, "multi_thread_hrv_processor_40s.py"])
