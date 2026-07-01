"""
Agentic RAG Assistant — Entry Point
Launches the Streamlit-based RAG assistant application.
"""

import os
import subprocess
import sys


def main():
    """
    Entrypoint to run the Streamlit app.
    Checks for the UI file and launches the Streamlit server.
    """
    print("\n🧠 Agentic RAG Assistant")
    print("=" * 40)

    # Locate the Streamlit app
    file_path = os.path.join("ui", "app.py")
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    # Launch Streamlit
    print(f"→ Starting Streamlit server...")
    cmd = [sys.executable, "-m", "streamlit", "run", file_path]

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n✓ Application stopped.")
    except Exception as e:
        print(f"✗ Error occurred: {e}")


if __name__ == "__main__":
    main()
