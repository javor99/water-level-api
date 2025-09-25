#!/usr/bin/env python3
"""
Fix the password verification function
"""

def fix_password_verification():
    """Fix the verify_password function."""
    
    print("🔧 Fixing password verification function...")
    
    # Read the current server file
    with open('water_level_server_with_municipalities.py', 'r') as f:
        content = f.read()
    
    # Fix the verify_password function
    old_function = '''def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))'''
    
    new_function = '''def verify_password(password: str, hashed: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed)'''
    
    if old_function in content:
        content = content.replace(old_function, new_function)
        print("✅ Fixed password verification function")
        print("   - Removed .encode('utf-8') from hashed parameter")
    else:
        print("❌ Could not find the verify_password function to fix")
        return False
    
    # Write the updated content back
    with open('water_level_server_with_municipalities.py', 'w') as f:
        f.write(content)
    
    print("🎉 Password verification function fixed!")
    return True

if __name__ == "__main__":
    fix_password_verification()
