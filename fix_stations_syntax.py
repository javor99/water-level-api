#!/usr/bin/env python3

# Read the file line by line
with open('water_level_server_with_municipalities.py', 'r') as f:
    lines = f.readlines()

new_lines = []
i = 0
while i < len(lines):
    line = lines[i]
    
    # Find the problematic function definition
    if line.strip() == "def get_stations():":
        # Replace this line with proper indentation for the GET request logic
        new_lines.append("    # GET request - list all stations\n")
        i += 1
        # Skip the docstring line
        if i < len(lines) and '"""Get all stations' in lines[i]:
            i += 1
        continue
    else:
        new_lines.append(line)
    
    i += 1

# Write the fixed content
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed the syntax error in stations function")
