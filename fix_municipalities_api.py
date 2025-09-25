#!/usr/bin/env python3
"""
Fix the municipalities INSERT statement in the API code
"""

import re

def fix_municipalities_api():
    """Fix the municipalities INSERT statement."""
    
    print("🔧 Fixing municipalities INSERT statement in API code...")
    
    # Read the current API file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Find and replace the incorrect INSERT statement
    old_insert = '''        cursor.execute("""
            INSERT INTO municipalities 
            (name, region, population, area_km2, description, created_by, updated_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, region, population, area_km2, description, 
              get_user_email_from_jwt(), get_user_email_from_jwt()))'''
    
    new_insert = '''        cursor.execute("""
            INSERT INTO municipalities 
            (name, region, population, area_km2, description, created_by, updated_by, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (name, region, population, area_km2, description, 
              get_user_email_from_jwt(), get_user_email_from_jwt(), 
              'CURRENT_TIMESTAMP', 'CURRENT_TIMESTAMP'))'''
    
    # Replace the incorrect INSERT
    if old_insert in content:
        content = content.replace(old_insert, new_insert)
        
        # Write the fixed content back
        with open('water_level_server_with_municipalities.py', 'w') as f:
            f.write(content)
        
        print("✅ Fixed municipalities INSERT statement in API code")
        print("✅ Added created_at and updated_at columns to INSERT")
        print("✅ Added CURRENT_TIMESTAMP values for created_at and updated_at")
    else:
        print("❌ Could not find the INSERT statement to fix")
        print("🔍 The INSERT statement might have been changed already")
    
    print("🎉 API code fix complete!")
    print("✅ You can now create municipalities without 500 errors")

if __name__ == "__main__":
    fix_municipalities_api()
