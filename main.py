import os
import sys
import subprocess

def main():
    """
    Entrypoint to run the Streamlit app.
    """
    print("Starting Agentic RAG Assistant...")
    
    # Ensure dependencies are installed (optional check)
    # print("Checking dependencies...")
    
    # Run the embeddings generation if vector store doesn't exist
    if not os.path.exists("faiss_index"):
        print("Vector store not found. Generating embeddings (this may take a moment)...")
        # We can import and run the generation, or just let the app handle it.
        # The app.py handles initialization, so we can just launch it.
    
    # Launch Streamlit
    file_path = os.path.join("ui", "app.py")
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    cmd = ["streamlit", "run", file_path]
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nStopping application.")
    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    main()
