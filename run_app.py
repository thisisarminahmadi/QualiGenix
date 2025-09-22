"""
QualiGenix Streamlit App Runner
"""

import subprocess
import sys
from pathlib import Path

def run_streamlit():
    """Run the Streamlit application"""
    print("🚀 Starting QualiGenix Streamlit Dashboard...")
    print("="*60)
    print("The dashboard will open in your browser at: http://localhost:8501")
    print("Press Ctrl+C to stop the server")
    print("="*60)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 QualiGenix dashboard stopped.")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

if __name__ == "__main__":
    run_streamlit() 