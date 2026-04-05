import sys
import os

# Add /app to path so pytest can find processing/, websocket_manager, etc.
sys.path.insert(0, "/app")
# Add /app/processing to path so pipeline.py can find segment_by_texture, etc.
sys.path.insert(0, "/app/processing")