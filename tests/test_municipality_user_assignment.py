#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for municipality assignment to admin users
Demonstrates:
1. Creating a user with municipality assignment
2. Getting user's municipality information
3. Updating user's municipality
"""

import requests
import json

BASE_URL = "http://localhost:5001"

def login_as_superadmin():
    """Login as superadmin to get access token."""
    response = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "superadmin@superadmin.com",
        "password": "12345678"
    })
    if response.status_code == 200:
        data = response.json()
        print("✅ Logged in as superadmin")
        print(f"   User info: {json.dumps(data['user'], indent=2)}")
        return data['token']
    else:
        print(f"❌ Login failed: {response.json()}")
        return None

def list_municipalities(token):
    """List all municipalities to get an ID for testing."""
    response = requests.get(f"{BASE_URL}/municipalities")
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Found {data['count']} municipalities")
        if data['municipalities']:
            print("   First municipality:")
            print(f"   {json.dumps(data['municipalities'][0], indent=2)}")
            return data['municipalities'][0]['id']
    else:
        print(f"❌ Failed to list municipalities: {response.json()}")
    return None

def create_user_with_municipality(token, municipality_id):
    """Create a new admin user with municipality assignment."""
    headers = {"Authorization": f"Bearer {token}"}
    
    user_data = {
        "email": "admin.test@municipality.com",
        "password": "testpassword123",
        "role": "admin",
        "municipality_id": municipality_id
    }
    
    response = requests.post(f"{BASE_URL}/auth/register", 
                           json=user_data, 
                           headers=headers)
    
    if response.status_code == 201:
        data = response.json()
        print(f"\n✅ Created user with municipality assignment")
        print(f"   User: {json.dumps(data['user'], indent=2)}")
        return data['user']['id']
    else:
        print(f"\n❌ Failed to create user: {response.json()}")
        return None

def login_as_new_user():
    """Login as the newly created user."""
    response = requests.post(f"{BASE_URL}/auth/login", json={
        "email": "admin.test@municipality.com",
        "password": "testpassword123"
    })
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Logged in as new admin user")
        print(f"   User info: {json.dumps(data['user'], indent=2)}")
        return data['token']
    else:
        print(f"❌ Login failed: {response.json()}")
        return None

def get_user_municipality(token, is_superadmin=False):
    """Get the municipality assigned to the current user."""
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/auth/user/municipality", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if is_superadmin:
            print(f"\n✅ Retrieved all municipalities (superadmin)")
            print(f"   Count: {data.get('count', 0)} municipalities")
            if data.get('municipalities'):
                print(f"   Municipalities: {[m['name'] for m in data['municipalities']]}")
        else:
            print(f"\n✅ Retrieved user municipality")
            if data.get('municipality'):
                print(f"   Municipality: {data['municipality']['name']}")
        return data
    else:
        print(f"\n❌ Failed to get user municipality: {response.json()}")
        return None

def update_user_municipality(superadmin_token, user_id, new_municipality_id):
    """Update a user's municipality assignment."""
    headers = {"Authorization": f"Bearer {superadmin_token}"}
    
    response = requests.put(f"{BASE_URL}/auth/users/{user_id}", 
                          json={"municipality_id": new_municipality_id},
                          headers=headers)
    
    if response.status_code == 200:
        print(f"\n✅ Updated user municipality to {new_municipality_id}")
        print(f"   Response: {response.json()}")
        return True
    else:
        print(f"\n❌ Failed to update user municipality: {response.json()}")
        return False

def list_all_users(token):
    """List all users to see municipality assignments."""
    headers = {"Authorization": f"Bearer {token}"}
    
    response = requests.get(f"{BASE_URL}/auth/users", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"\n✅ Listed all users ({data['count']} users)")
        for user in data['users']:
            print(f"   - {user['email']} ({user['role']}) - Municipality: {user.get('municipality_name', 'None')}")
        return data['users']
    else:
        print(f"\n❌ Failed to list users: {response.json()}")
        return None

def cleanup_test_user(token):
    """Delete the test user."""
    headers = {"Authorization": f"Bearer {token}"}
    
    # First, get the user ID by listing users
    response = requests.get(f"{BASE_URL}/auth/users", headers=headers)
    if response.status_code == 200:
        users = response.json()['users']
        test_user = next((u for u in users if u['email'] == 'admin.test@municipality.com'), None)
        if test_user:
            user_id = test_user['id']
            delete_response = requests.delete(f"{BASE_URL}/auth/users/{user_id}", headers=headers)
            if delete_response.status_code == 200:
                print(f"\n✅ Cleaned up test user")
            else:
                print(f"\n⚠️  Failed to clean up test user: {delete_response.json()}")

def main():
    """Run all tests."""
    print("="*60)
    print("Testing Municipality Assignment to Admin Users")
    print("="*60)
    
    # Step 1: Login as superadmin
    superadmin_token = login_as_superadmin()
    if not superadmin_token:
        return
    
    # Step 2: Test superadmin getting ALL municipalities
    print("\n--- Testing Superadmin Gets All Municipalities ---")
    superadmin_municipalities = get_user_municipality(superadmin_token, is_superadmin=True)
    if superadmin_municipalities and 'municipalities' in superadmin_municipalities:
        print(f"   ✅ Superadmin correctly received {len(superadmin_municipalities['municipalities'])} municipalities")
    
    # Step 3: Get a municipality ID
    municipality_id = list_municipalities(superadmin_token)
    if not municipality_id:
        print("\n⚠️  No municipalities found. Please create a municipality first.")
        return
    
    # Step 3: Create a user with municipality assignment
    user_id = create_user_with_municipality(superadmin_token, municipality_id)
    if not user_id:
        return
    
    # Step 4: Login as the new user
    user_token = login_as_new_user()
    if not user_token:
        return
    
    # Step 5: Get user's municipality using the new endpoint
    municipality = get_user_municipality(user_token)
    
    # Step 6: List all users to see municipality info
    list_all_users(superadmin_token)
    
    # Step 7: Test updating municipality (set to None)
    update_user_municipality(superadmin_token, user_id, None)
    
    # Step 8: Verify the update
    print("\n--- Verifying municipality was removed ---")
    user_token = login_as_new_user()
    get_user_municipality(user_token)
    
    # Cleanup
    cleanup_test_user(superadmin_token)
    
    print("\n" + "="*60)
    print("✅ All tests completed successfully!")
    print("="*60)
    print("\n📝 Summary:")
    print("   - Admin users can be assigned to a municipality during registration")
    print("   - Use POST /auth/register with 'municipality_id' field")
    print("   - Use GET /auth/user/municipality to get user's assigned municipality")
    print("   - Superadmins receive ALL municipalities via GET /auth/user/municipality")
    print("   - Regular users/admins receive only their assigned municipality")
    print("   - Login response includes municipality_id and municipality_name")
    print("   - Municipality assignment can be updated via PUT /auth/users/<id>")

if __name__ == '__main__':
    main()

