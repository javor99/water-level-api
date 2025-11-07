#!/usr/bin/env python3
"""
Automated Route Extractor
Extracts routes from water_level_server_with_municipalities.py
and populates the blueprint files.
"""

import re

def read_file(filepath):
    """Read file contents."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

def extract_route_handler(content, start_line):
    """Extract a complete route handler starting from a given line."""
    lines = content.split('\n')
    if start_line >= len(lines):
        return None
    
    # Find the start of the function (may have multiple decorators)
    current = start_line - 1  # Convert to 0-indexed
    
    # Collect decorators
    decorators = []
    while current >= 0 and (lines[current].strip().startswith('@') or lines[current].strip() == ''):
        if lines[current].strip().startswith('@'):
            decorators.insert(0, lines[current])
        current += 1
        if current >= len(lines):
            break
        if lines[current].strip().startswith('def '):
            break
    
    # Find function definition
    func_start = current
    if func_start >= len(lines) or not lines[func_start].strip().startswith('def '):
        return None
    
    # Extract function body
    indent_level = len(lines[func_start]) - len(lines[func_start].lstrip())
    func_lines = [lines[func_start]]
    
    current = func_start + 1
    while current < len(lines):
        line = lines[current]
        
        # Stop at next function or decorator at same level
        if line.strip() and not line.startswith(' ' * (indent_level + 1)):
            if line.strip().startswith('@app.route') or line.strip().startswith('def ') or line.strip().startswith('if __name__'):
                break
        
        func_lines.append(line)
        current += 1
    
    # Combine decorators and function
    full_handler = '\n'.join(decorators + func_lines)
    return full_handler.rstrip()

# Read the monolithic file
print("📖 Reading water_level_server_with_municipalities.py...")
content = read_file('water_level_server_with_municipalities.py')

# Route mappings (route_name: line_number)
route_map = {
    'auth': {
        275: 'register_user',
        362: 'login_user',
        437: 'verify_token',
        449: 'list_users',
        489: 'get_user_municipality',
        589: 'update_user',
        672: 'delete_user'
    },
    'municipalities': {
        711: 'list_municipalities',
        755: 'get_municipality',
        816: 'create_municipality',
        878: 'update_municipality',
        941: 'delete_municipality',
        977: 'assign_stations_to_municipality',
        1036: 'get_stations_by_municipalities',
        1110: 'get_weather_stations_by_municipalities'
    },
    'stations': {
        1197: 'get_weather_station',
        1205: 'get_stations',
        1240: 'create_station',
        1361: 'get_station',
        1397: 'delete_station',
        1671: 'get_station_minmax',
        1724: 'update_station_minmax',
        1786: 'update_bulk_station_minmax'
    },
    'predictions': {
        1435: 'get_water_levels',
        1498: 'get_station_water_levels',
        1537: 'get_predictions',
        1582: 'get_station_predictions',
        1624: 'get_station_past_predictions'
    },
    'subscriptions': {
        1882: 'subscribe_to_station',
        1947: 'unsubscribe_from_station',
        1995: 'get_user_subscriptions'
    }
}

print(f"\n🔍 Found {sum(len(v) for v in route_map.values())} routes to extract")
print("\n" + "="*60)

for category, routes in route_map.items():
    print(f"\n📁 Processing {category.upper()} routes...")
    
    for line_num, func_name in routes.items():
        handler = extract_route_handler(content, line_num)
        if handler:
            # Replace @app.route with @{category}_bp.route
            bp_name = f"{category}_bp" if category != 'predictions' else 'predictions_bp'
            if category == 'subscriptions':
                bp_name = 'subscriptions_bp'
            
            handler = handler.replace('@app.route', f'@{bp_name}.route')
            print(f"  ✓ Extracted {func_name} ({len(handler.split(chr(10)))} lines)")
        else:
            print(f"  ✗ Failed to extract {func_name} at line {line_num}")

print("\n" + "="*60)
print("\n📝 To complete the migration:")
print("1. Review the extracted code above")
print("2. Manually copy each route handler to its blueprint file")
print("3. Add necessary imports to each blueprint")
print("4. Test each endpoint")
print("\n💡 Tip: Start with simple routes (GET endpoints) first")



