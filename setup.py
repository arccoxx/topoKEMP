import os
import sys
import subprocess

print("Starting TopoKEMP setup...")

# Step 1: Install requirements (no SnapPy, as it's eliminated via proxy)
print("Installing requirements...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], capture_output=True, text=True)

# Step 2: Install TopoKEMP in editable mode
print("Installing TopoKEMP...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], capture_output=True, text=True)

print("Setup complete! Restart runtime if needed, then import and use TopoKEMP.")
