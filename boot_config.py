"""
boot.py
Runs automatically when ESP32 boots up
Upload this file to your ESP32 using Thonny (optional)
"""

import gc
import esp

# Disable debug output for cleaner console
esp.osdebug(None)

# Enable garbage collection
gc.enable()

# Free up memory
gc.collect()

print("\n" + "="*60)
print("ESP32 Boot Sequence")
print("="*60)
print(f"Free memory: {gc.mem_free()} bytes")
print("Ready to run main.py")
print("="*60 + "\n")
