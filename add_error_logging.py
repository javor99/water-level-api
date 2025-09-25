#!/usr/bin/env python3
"""
Add comprehensive error logging to the water level server
"""

import re

def add_error_logging():
    """Add error logging to the server."""
    
    print("🔧 Adding comprehensive error logging to server...")
    
    # Read the current server file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Add logging import if not present
    if 'import logging' not in content:
        content = content.replace(
            'import os\nimport sqlite3',
            'import os\nimport logging\nimport sqlite3'
        )
    
    # Add logging configuration
    logging_config = '''
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

'''
    
    # Add logging config after imports
    if 'logging.basicConfig' not in content:
        content = content.replace(
            'from flask_cors import CORS',
            'from flask_cors import CORS' + logging_config
        )
    
    # Add error logging to municipality creation
    old_municipality_create = '''@app.route("/municipalities", methods=["POST"])
def create_municipality():
    """Create a new municipality (superadmin only)."""
    try:'''
    
    new_municipality_create = '''@app.route("/municipalities", methods=["POST"])
def create_municipality():
    """Create a new municipality (superadmin only)."""
    logger.info("Municipality creation request received")
    try:'''
    
    if old_municipality_create in content:
        content = content.replace(old_municipality_create, new_municipality_create)
        print("✅ Added logging to municipality creation")
    
    # Add error logging to the catch block
    old_catch = '''    except Exception as e:
        return jsonify({"error": f"Failed to create municipality: {str(e)}"}), 500'''
    
    new_catch = '''    except Exception as e:
        logger.error(f"Municipality creation failed: {str(e)}", exc_info=True)
        return jsonify({"error": f"Failed to create municipality: {str(e)}"}), 500'''
    
    if old_catch in content:
        content = content.replace(old_catch, new_catch)
        print("✅ Added error logging to municipality creation catch block")
    
    # Add error logging to station creation
    old_station_create = '''@app.route('/stations', methods=['POST'])
@require_role('superadmin')
def create_station():
    """Create new station - superadmin required."""
    try:'''
    
    new_station_create = '''@app.route('/stations', methods=['POST'])
@require_role('superadmin')
def create_station():
    """Create new station - superadmin required."""
    logger.info("Station creation request received")
    try:'''
    
    if old_station_create in content:
        content = content.replace(old_station_create, new_station_create)
        print("✅ Added logging to station creation")
    
    # Add error logging to login
    old_login = '''@app.route('/auth/login', methods=['POST'])
def login():
    """Login endpoint."""
    try:'''
    
    new_login = '''@app.route('/auth/login', methods=['POST'])
def login():
    """Login endpoint."""
    logger.info("Login request received")
    try:'''
    
    if old_login in content:
        content = content.replace(old_login, new_login)
        print("✅ Added logging to login")
    
    # Add error logging to all catch blocks
    content = re.sub(
        r'except Exception as e:\s*\n\s*return jsonify\(\{"error": f"([^"]+)": \{str\(e\)\}"\}\), 500',
        r'except Exception as e:\n        logger.error(f"\1: {str(e)}", exc_info=True)\n        return jsonify({"error": f"\1: {str(e)}"}), 500',
        content
    )
    
    # Write the updated content back
    with open('water_level_server_with_municipalities.py', 'w') as f:
        f.write(content)
    
    print("🎉 Error logging added to server!")
    print("✅ All errors will now be logged to server.log")
    print("✅ You can monitor errors with: tail -f server.log")

if __name__ == "__main__":
    add_error_logging()
