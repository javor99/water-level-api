#!/usr/bin/env python3

# Read the current server file
with open('water_level_server_with_municipalities.py', 'r') as f:
    lines = f.readlines()

# Find and fix the malformed stations function
new_lines = []
in_stations_function = False
for i, line in enumerate(lines):
    if "@app.route('/stations', methods=['GET', 'POST'])" in line:
        in_stations_function = True
        new_lines.append(line)
    elif in_stations_function and line.strip() == "# GET request - list stations":
        # This is where we need to add the return for the POST case and fix the GET logic
        new_lines.append("    \n")  # Add some space
        new_lines.append("    # GET request - list stations\n")
    elif in_stations_function and line.strip().startswith("def get_stations():"):
        # This should not be a separate function definition, remove the def line
        continue
    elif in_stations_function and line.startswith("@app.route"):
        # We've reached the next route, stop processing
        in_stations_function = False
        new_lines.append(line)
    else:
        new_lines.append(line)

# Write the fixed content
with open('water_level_server_with_municipalities.py', 'w') as f:
    f.writelines(new_lines)

print("Fixed the stations route structure")
