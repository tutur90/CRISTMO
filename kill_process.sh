#!/bin/bash

# Kill all processes using NVIDIA devices
echo "Finding processes using NVIDIA devices..."

# Get PIDs using fuser
pids=$(sudo fuser /dev/nvidia* 2>/dev/null | tr ' ' '\n' | grep -E '^[0-9]+$' | sort -u)

# Check if any processes were found
if [ -z "$pids" ]; then
    echo "No processes found using NVIDIA devices."
    exit 0
fi

echo "Found the following PIDs using NVIDIA devices:"
echo "$pids"
echo ""
echo "Killing processes..."

# Kill each process
for pid in $pids; do
    if [ -n "$pid" ]; then
        # Get process info
        proc_info=$(ps -p "$pid" -o comm= 2>/dev/null)
        echo "Killing PID $pid ($proc_info)"
        sudo kill -9 "$pid" 2>/dev/null
        if [ $? -eq 0 ]; then
            echo "  ✓ Process $pid killed successfully"
        else
            echo "  ✗ Failed to kill process $pid"
        fi
    fi
done

echo ""
echo "Done. Verifying..."
remaining=$(sudo fuser /dev/nvidia* 2>/dev/null)
if [ -z "$remaining" ]; then
    echo "✓ All NVIDIA device processes have been terminated."
else
    echo "⚠ Some processes may still be running:"
    sudo fuser -v /dev/nvidia* 2>/dev/null
fi