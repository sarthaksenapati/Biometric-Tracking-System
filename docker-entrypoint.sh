#!/bin/bash
# Entrypoint script for the tracker service
# Starts virtual display (Xvfb) and then runs the tracker

set -e

# Start Xvfb (virtual framebuffer) for OpenCV
Xvfb :99 -screen 0 640x480x24 -ac +extension GLX +render -noreset &
XVFB_PID=$!

# Wait for Xvfb to be ready
echo "Waiting for Xvfb to start..."
for i in {1..10}; do
    if xdpyinfo -display :99 >/dev/null 2>&1; then
        echo "Xvfb is ready!"
        break
    fi
    sleep 0.5
done

# Initialize database if using PostgreSQL
if [ "$USE_DATABASE" = "true" ]; then
    echo "Initializing database..."
    python -c "from db.connection import init_db; init_db()" || echo "DB init failed, continuing anyway..."
fi

# Trap signals to properly shutdown
trap "echo 'Stopping...'; kill $XVFB_PID; exit" SIGINT SIGTERM

# Execute the command passed to the container
echo "Starting tracker..."
exec "$@"
