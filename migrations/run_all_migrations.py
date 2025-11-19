#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run all pending database migrations in order.

This script runs all migration files in the migrations directory
in numerical order (001, 002, 003, etc.)
"""

import os
import sys
import importlib.util
from pathlib import Path


def get_migration_files():
    """Get all migration files sorted by number."""
    migrations_dir = Path(__file__).parent
    migration_files = []
    
    for file in sorted(migrations_dir.glob("*.py")):
        if file.name.startswith("__") or file.name == "run_all_migrations.py":
            continue
        
        # Extract migration number from filename (e.g., "001_add_..." -> 1)
        try:
            migration_num = int(file.stem.split("_")[0])
            migration_files.append((migration_num, file))
        except ValueError:
            continue
    
    return sorted(migration_files)


def run_migration(migration_file):
    """Run a single migration file."""
    print(f"\n{'='*70}")
    print(f"Running: {migration_file.name}")
    print(f"{'='*70}\n")
    
    spec = importlib.util.spec_from_file_location("migration", migration_file)
    module = importlib.util.module_from_spec(spec)
    
    try:
        spec.loader.exec_module(module)
        
        if hasattr(module, 'migrate'):
            success = module.migrate()
            return success
        else:
            print(f"❌ Migration {migration_file.name} does not have a migrate() function")
            return False
    except Exception as e:
        print(f"❌ Error running migration {migration_file.name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all migrations."""
    print("=" * 70)
    print("🔄 Running All Database Migrations")
    print("=" * 70)
    print()
    
    migration_files = get_migration_files()
    
    if not migration_files:
        print("⚠️  No migration files found")
        return 0
    
    print(f"Found {len(migration_files)} migration(s):")
    for num, file in migration_files:
        print(f"  {num:03d}: {file.name}")
    print()
    
    # Ask for confirmation
    response = input("Do you want to run all migrations? (yes/no): ")
    if response.lower() != 'yes':
        print("❌ Migration cancelled")
        return 1
    
    print()
    print("Starting migrations...")
    print()
    
    success_count = 0
    failed_migrations = []
    
    for num, migration_file in migration_files:
        success = run_migration(migration_file)
        if success:
            success_count += 1
        else:
            failed_migrations.append(migration_file.name)
            print(f"\n⚠️  Migration {migration_file.name} failed!")
            response = input("Continue with remaining migrations? (yes/no): ")
            if response.lower() != 'yes':
                break
    
    print()
    print("=" * 70)
    print("📊 Migration Summary")
    print("=" * 70)
    print(f"✅ Successful: {success_count}/{len(migration_files)}")
    
    if failed_migrations:
        print(f"❌ Failed: {len(failed_migrations)}")
        print("   Failed migrations:")
        for migration in failed_migrations:
            print(f"     - {migration}")
        return 1
    else:
        print("✅ All migrations completed successfully!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

