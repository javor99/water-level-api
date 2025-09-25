#!/usr/bin/env python3
"""Fix station creation to make name optional"""

# Read the file
with open('water_level_server_with_municipalities.py', 'r') as f:
    content = f.read()

# Fix the name field to be optional
old_line = '        name = data["name"].strip()'
new_line = '        name = data.get("name", "").strip()'

if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ Fixed name field to be optional")
else:
    print("❌ Could not find the line to fix")

# Write the fixed content back
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.write(content)

print("✅ Station creation now accepts optional name field")
