#!/usr/bin/env python3
"""
Script to split the monolithic water_level_server_with_municipalities.py
into modular route blueprints.
"""

import re

def extract_routes_by_prefix(content, prefix):
    """Extract all routes that start with a given prefix."""
    # Find all route definitions
    pattern = rf"(@app\.route\('{re.escape(prefix)}[^']*'[^\n]*\n(?:@[^\n]+\n)*def [^(]+\([^)]*\):[^\n]*\n(?:(?!@app\.route|if __name__).*\n)*)"
    matches = re.finditer(pattern, content, re.MULTILINE)
    routes = []
    for match in matches:
        routes.append(match.group(0))
    return routes

# Read the monolithic file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Extract import statements
imports_end = content.find('def require_auth(f):')
imports_section = content[:imports_end]

# Extract routes by category
auth_start = content.find('# ===== USER AUTHENTICATION ENDPOINTS =====')
auth_end = content.find('# ===== MUNICIPALITY ENDPOINTS =====')
auth_section = content[auth_start:auth_end] if auth_start != -1 and auth_end != -1 else ""

munic_start = content.find('# ===== MUNICIPALITY ENDPOINTS =====')
munic_end = content.find("@app.route('/')")
munic_section = content[munic_start:munic_end] if munic_start != -1 and munic_end != -1 else ""

print(f"Found auth section: {len(auth_section)} chars")
print(f"Found municipality section: {len(munic_section)} chars")

# Count route decorators
auth_routes = auth_section.count('@app.route')
munic_routes = munic_section.count('@app.route')

print(f"\nAuth routes: {auth_routes}")
print(f"Municipality routes: {munic_routes}")



