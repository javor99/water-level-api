"""User model and operations."""
from app.utils.database import get_db_connection
from app.utils.password import hash_password


def init_user_table():
    """Initialize the users table with roles if it doesn't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            municipality_id INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            created_by INTEGER,
            FOREIGN KEY (created_by) REFERENCES users (id),
            FOREIGN KEY (municipality_id) REFERENCES municipalities (id)
        )
    ''')
    
    conn.commit()
    conn.close()


def create_default_users():
    """Create default admin and superadmin users."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if users already exist
    cursor.execute('SELECT COUNT(*) FROM users')
    user_count = cursor.fetchone()[0]
    
    if user_count == 0:
        # Create superadmin user
        superadmin_password_hash = hash_password('12345678')
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, created_by)
            VALUES (?, ?, ?, ?)
        ''', ('superadmin@superadmin.com', superadmin_password_hash, 'superadmin', None))
        
        superadmin_id = cursor.lastrowid
        
        # Create admin user
        admin_password_hash = hash_password('12345678')
        cursor.execute('''
            INSERT INTO users (email, password_hash, role, created_by)
            VALUES (?, ?, ?, ?)
        ''', ('admin@admin.com', admin_password_hash, 'admin', superadmin_id))
        
        conn.commit()
        print("Default users created:")
        print("  superadmin@superadmin.com (password: 12345678)")
        print("  admin@admin.com (password: 12345678)")
    
    conn.close()



