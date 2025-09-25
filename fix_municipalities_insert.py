#!/usr/bin/env python3
"""
Fix the municipalities INSERT statement to match the number of columns and values
"""

def fix_municipalities_insert():
    """Fix the INSERT statement in the municipalities endpoint."""
    
    print("🔧 Fixing municipalities INSERT statement...")
    
    # Read the current server file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Find and fix the INSERT statement
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
              datetime.now().isoformat(), datetime.now().isoformat()))'''
    
    if old_insert in content:
        content = content.replace(old_insert, new_insert)
        print("✅ Fixed municipalities INSERT statement")
        print("   - Added created_at and updated_at to column list")
        print("   - Added datetime.now() values for created_at and updated_at")
    else:
        print("❌ Could not find the INSERT statement to fix")
        return False
    
    # Write the updated content back
    with open('water_level_server_with_municipalities.py', 'w') as f:
        f.write(content)
    
    print("🎉 Municipalities INSERT statement fixed!")
    print("✅ Now 9 columns match 9 values")
    return True

if __name__ == "__main__":
    fix_municipalities_insert()
