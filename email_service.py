#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email Service for Water Level Alerts
Handles sending email notifications for water level predictions
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        """Initialize email service with SMTP configuration."""
        # Email configuration from environment variables
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '587'))
        self.smtp_username = os.environ.get('SMTP_USERNAME', '')
        self.smtp_password = os.environ.get('SMTP_PASSWORD', '')
        self.from_email = os.environ.get('FROM_EMAIL', self.smtp_username)
        self.from_name = os.environ.get('FROM_NAME', 'Water Level Alert System')
        
        # Validate configuration
        if not self.smtp_username or not self.smtp_password:
            logger.warning("⚠️  Email service not configured - SMTP credentials missing")
            logger.warning("Set SMTP_USERNAME and SMTP_PASSWORD environment variables")
            self.enabled = False
        else:
            self.enabled = True
            logger.info(f"✅ Email service configured for {self.from_email}")

    def send_water_level_alert(self, user_email, station_name, station_id, 
                             current_prediction, max_level, threshold_percentage=0.9):
        """
        Send water level alert email to user.
        
        Args:
            user_email (str): Recipient email address
            station_name (str): Name of the water level station
            station_id (str): Station ID
            current_prediction (float): Current predicted water level
            max_level (float): Maximum historical water level
            threshold_percentage (float): Alert threshold (default 0.9 = 90%)
        """
        if not self.enabled:
            logger.warning(f"📧 Email service disabled - would send alert to {user_email}")
            return False

        try:
            # Calculate threshold level
            threshold_level = max_level * threshold_percentage
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = user_email
            msg['Subject'] = f"🚨 Water Level Alert - {station_name}"
            
            # Create email body
            body = f"""
🚨 WATER LEVEL ALERT 🚨

Station: {station_name} (ID: {station_id})
Current Prediction: {current_prediction:.2f} meters
Maximum Historical Level: {max_level:.2f} meters
Alert Threshold: {threshold_percentage*100:.0f}% ({threshold_level:.2f} meters)

⚠️  WARNING: The predicted water level ({current_prediction:.2f}m) has exceeded 
the alert threshold of {threshold_percentage*100:.0f}% of the maximum historical level.

This alert was triggered at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Please take appropriate precautions and monitor the situation closely.

---
Water Level Alert System
Automated notification service
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                text = msg.as_string()
                server.sendmail(self.from_email, user_email, text)
            
            logger.info(f"�� Alert email sent successfully to {user_email} for station {station_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send alert email to {user_email}: {str(e)}")
            return False

    def send_subscription_confirmation(self, user_email, station_name, station_id):
        """
        Send subscription confirmation email.
        
        Args:
            user_email (str): Recipient email address
            station_name (str): Name of the water level station
            station_id (str): Station ID
        """
        if not self.enabled:
            logger.warning(f"�� Email service disabled - would send confirmation to {user_email}")
            return False

        try:
            msg = MIMEMultipart()
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = user_email
            msg['Subject'] = f"✅ Subscription Confirmed - {station_name}"
            
            body = f"""
✅ SUBSCRIPTION CONFIRMED

You have successfully subscribed to water level alerts for:
Station: {station_name} (ID: {station_id})

You will receive email notifications when the predicted water level exceeds 
90% of the maximum historical level for this station.

Subscription activated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

To unsubscribe, please contact the system administrator or use the unsubscribe API endpoint.

---
Water Level Alert System
Automated notification service
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                text = msg.as_string()
                server.sendmail(self.from_email, user_email, text)
            
            logger.info(f"📧 Confirmation email sent to {user_email} for station {station_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to send confirmation email to {user_email}: {str(e)}")
            return False

    def test_email_connection(self):
        """Test email service connection."""
        if not self.enabled:
            logger.warning("📧 Email service not enabled - cannot test connection")
            return False

        try:
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
            logger.info("✅ Email service connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Email service connection test failed: {str(e)}")
            return False

# Global email service instance
email_service = EmailService()

def send_water_level_alert(user_email, station_name, station_id, 
                          current_prediction, max_level, threshold_percentage=0.9):
    """Convenience function to send water level alert."""
    return email_service.send_water_level_alert(
        user_email, station_name, station_id, 
        current_prediction, max_level, threshold_percentage
    )

def send_subscription_confirmation(user_email, station_name, station_id):
    """Convenience function to send subscription confirmation."""
    return email_service.send_subscription_confirmation(user_email, station_name, station_id)

def test_email_connection():
    """Convenience function to test email connection."""
    return email_service.test_email_connection()

if __name__ == "__main__":
    # Test the email service
    print("🧪 Testing Email Service...")
    print("=" * 50)
    
    # Test connection
    if test_email_connection():
        print("✅ Email service is working correctly")
        
        # Test sending a sample alert (uncomment to test)
        # send_water_level_alert(
        #     "test@example.com",
        #     "Test Station",
        #     "TEST001",
        #     1.5,
        #     2.0,
        #     0.9
        # )
    else:
        print("❌ Email service configuration issue")
        print("Please set the following environment variables:")
        print("- SMTP_SERVER (default: smtp.gmail.com)")
        print("- SMTP_PORT (default: 587)")
        print("- SMTP_USERNAME")
        print("- SMTP_PASSWORD")
        print("- FROM_EMAIL (optional, defaults to SMTP_USERNAME)")
        print("- FROM_NAME (optional, defaults to 'Water Level Alert System')")
