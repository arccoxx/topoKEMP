import os
import sys
import subprocess

print("Starting TopoKEMP setup...")

# Step 1: Install SnapPy (knot theory version) via user pip
print("Installing SnapPy...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', '--user', 'snappy', 'snappy_15_knots'], capture_output=True, text=True)

# Step 2: Set Colab path for SnapPy (add user site-packages)
import site
user_site = site.getusersitepackages()
sys.path.append(user_site)
print("Added SnapPy path:", user_site)

# Test SnapPy import
try:
    import snappy
    print("SnapPy version:", snappy.__version__)
    knot = snappy.Link(braid=[1, -2, 3])
    print("SnapPy test successful: crossings =", knot.crossing_number())
except Exception as e:
    print("SnapPy test failed:", e)
    sys.exit(1)  # Stop if failed

# Step 3: Install other requirements
print("Installing requirements...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'], capture_output=True, text=True)

# Step 4: Install TopoKEMP in editable mode
print("Installing TopoKEMP...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-e', '.'], capture_output=True, text=True)

print("Setup complete! Restart runtime if needed, then import and use TopoKEMP.")
