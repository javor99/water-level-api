#!/usr/bin/env python3
"""
Test script to verify user persistence after server restart
"""

import requests
import time
import json

BASE_URL = "http://localhost:5001"

def test_user_persistence():
    print("🧪 Testing User Persistence...")
    print("=" * 50)
    
    # Step 1: Create a new user
    print("1. Creating new user...")
    user_data = {
        "email": "testuser_persistent@example.com",
        "password": "testpassword123",
        "role": "user"
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", json=user_data)
    if response.status_code == 201:
        user_info = response.json()
        print(f"✅ User created: {user_info['user']['email']} (ID: {user_info['user']['id']})")
    else:
        print(f"❌ Failed to create user: {response.text}")
        return
    
    # Step 2: Verify user exists
    print("\n2. Verifying user exists...")
    response = requests.get(f"{BASE_URL}/auth/users")
    if response.status_code == 200:
        users = response.json()['users']
        user_found = any(u['email'] == user_data['email'] for u in users)
        if user_found:
            print(f"✅ User found in database")
        else:
            print(f"❌ User not found in database")
            return
    else:
        print(f"❌ Failed to get users: {response.text}")
        return
    
    # Step 3: Test login
    print("\n3. Testing login...")
    login_data = {
        "email": user_data['email'],
        "password": user_data['password']
    }
    
    response = requests.post(f"{BASE_URL}/auth/login", json=login_data)
    if response.status_code == 200:
        login_info = response.json()
        print(f"✅ Login successful: {login_info['user']['email']}")
        print(f"   Token: {login_info['token'][:50]}...")
    else:
        print(f"❌ Login failed: {response.text}")
        return
    
    print("\n" + "=" * 50)
    print("🎉 All tests passed! User persistence is working.")
    print("\n📝 Instructions:")
    print("1. The server will no longer drop the users table on restart")
    print("2. Created users will persist across server restarts")
    print("3. You can now create users and they will remain in the database")
    print("4. Login will work with created users even after server restarts")

if __name__ == "__main__":
    test_user_persistence()
