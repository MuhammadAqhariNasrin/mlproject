import sys
import os

# Ensure src/ is discoverable
sys.path.append(os.path.dirname(__file__))

from app import app as application  # Flask app variable is "app"
