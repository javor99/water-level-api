#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Email Service for Water Level Alerts
Handles sending email notifications for water level predictions
Configured for DTU SMTP server (smtp.ait.dtu.dk)
"""

import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formataddr
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmailService:
    def __init__(self):
        """Initialize email service with DTU SMTP configuration."""
        # DTU SMTP Configuration (no authentication required)
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.ait.dtu.dk')
        self.smtp_port = int(os.environ.get('SMTP_PORT', '25'))
        self.from_email = os.environ.get('FROM_EMAIL', 'aquamonitor@dtu.dk')
        self.from_name = os.environ.get('FROM_NAME', 'Aqua Monitor')
        
        # Default recipients (can be overridden in method calls)
        default_receivers = os.environ.get('DEFAULT_RECEIVERS', 'aquamonitor@dtu.dk, s232467@student.dtu.dk')
        self.default_receivers = [email.strip() for email in default_receivers.split(',')]
        
        # DTU SMTP doesn't require authentication
        self.use_auth = os.environ.get('SMTP_USE_AUTH', 'false').lower() == 'true'
        self.smtp_username = os.environ.get('SMTP_USERNAME', '')
        self.smtp_password = os.environ.get('SMTP_PASSWORD', '')
        
        self.enabled = True
        logger.info(f"✅ Email service configured for {self.from_email} via {self.smtp_server}:{self.smtp_port}")
        logger.info(f"📧 Default receivers: {', '.join(self.default_receivers)}")

    def send_water_level_alert(self, user_email, station_name, station_id, 
                             current_prediction, min_level, max_level, threshold_percentage=0.9):
        """
        Send water level alert email to user and default receivers.
        
        Args:
            user_email (str): Recipient email address (also sent to default receivers)
            station_name (str): Name of the water level station
            station_id (str): Station ID
            current_prediction (float): Current predicted water level in cm
            min_level (float): Minimum historical water level in cm
            max_level (float): Maximum historical water level in cm
            threshold_percentage (float): Alert threshold (default 0.9 = 90% between min and max)
        """
        if not self.enabled:
            logger.warning(f"📧 Email service disabled - would send alert to {user_email}")
            return False

        try:
            # Calculate threshold level as percentage between min and max
            threshold_level = min_level + (max_level - min_level) * threshold_percentage
            
            # Combine recipients: user_email + default receivers
            all_recipients = list(set([user_email] + self.default_receivers))
            recipients_str = ', '.join(all_recipients)
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = formataddr((self.from_name, self.from_email))
            msg['To'] = recipients_str
            msg['Subject'] = f"🚨 Water Level Alert - {station_name}"
            
            # Create HTML email body
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="background-color: #ff4444; color: white; padding: 20px; border-radius: 5px;">
                    <h2 style="margin: 0;">🚨 WATER LEVEL ALERT 🚨</h2>
                </div>
                
                <div style="padding: 20px; background-color: #f9f9f9; margin-top: 20px; border-radius: 5px;">
                    <h3 style="color: #ff4444;">Station Information</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Station:</td>
                            <td style="padding: 8px;">{station_name} (ID: {station_id})</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Current Prediction:</td>
                            <td style="padding: 8px;"><strong style="color: #ff4444;">{current_prediction:.2f} cm</strong></td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Historical Range:</td>
                            <td style="padding: 8px;">{min_level:.2f} cm (min) - {max_level:.2f} cm (max)</td>
                        </tr>
                        <tr>
                            <td style="padding: 8px; font-weight: bold;">Alert Threshold:</td>
                            <td style="padding: 8px;">{threshold_percentage*100:.0f}% between min and max = {threshold_level:.2f} cm</td>
                        </tr>
                    </table>
                </div>
                
                <div style="padding: 20px; background-color: #fff3cd; margin-top: 20px; border-left: 5px solid #ffa500; border-radius: 5px;">
                    <p style="margin: 0;"><strong>⚠️ WARNING:</strong> The predicted water level ({current_prediction:.2f} cm) has exceeded 
                    the alert threshold of {threshold_percentage*100:.0f}% between the minimum and maximum historical levels.</p>
                </div>
                
                <div style="padding: 20px; margin-top: 20px;">
                    <p><strong>Alert triggered at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p>Please take appropriate precautions and monitor the situation closely.</p>
                </div>
                
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                
                <div style="padding: 10px; color: #666; font-size: 12px;">
                    <p style="margin: 0;"><strong>Aqua Monitor</strong></p>
                    <p style="margin: 5px 0;">Automated Water Level Alert System</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email using DTU SMTP (no authentication needed)
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.ehlo()
            
            # Only use authentication if required
            if self.use_auth and self.smtp_username and self.smtp_password:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
            
            server.sendmail(self.from_email, all_recipients, msg.as_string())
            server.quit()
            
            logger.info(f"📧 Alert email sent successfully to {recipients_str} for station {station_name}")
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
            logger.warning(f"📧 Email service disabled - would send confirmation to {user_email}")
            return False

        try:
            # Combine recipients: user_email + default receivers
            all_recipients = list(set([user_email] + self.default_receivers))
            recipients_str = ', '.join(all_recipients)
            
            msg = MIMEMultipart()
            msg['From'] = formataddr((self.from_name, self.from_email))
            msg['To'] = recipients_str
            msg['Subject'] = f"✅ Subscription Confirmed - {station_name}"
            
            # Create HTML email body
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
                <div style="background-color: #28a745; color: white; padding: 20px; border-radius: 5px;">
                    <h2 style="margin: 0;">✅ SUBSCRIPTION CONFIRMED</h2>
                </div>
                
                <div style="padding: 20px; background-color: #f9f9f9; margin-top: 20px; border-radius: 5px;">
                    <p>You have successfully subscribed to water level alerts for:</p>
                    <h3 style="color: #28a745;">{station_name} (ID: {station_id})</h3>
                </div>
                
                <div style="padding: 20px; margin-top: 20px;">
                    <p>You will receive email notifications when the predicted water level exceeds 
                    90% of the maximum historical level for this station.</p>
                    
                    <p><strong>Subscription activated at:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    
                    <p>To unsubscribe, please contact the system administrator or use the unsubscribe API endpoint.</p>
                </div>
                
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                
                <div style="padding: 10px; color: #666; font-size: 12px;">
                    <p style="margin: 0;"><strong>Aqua Monitor</strong></p>
                    <p style="margin: 5px 0;">Automated Water Level Alert System</p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email using DTU SMTP
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.ehlo()
            
            if self.use_auth and self.smtp_username and self.smtp_password:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
            
            server.sendmail(self.from_email, all_recipients, msg.as_string())
            server.quit()
            
            logger.info(f"📧 Confirmation email sent to {recipients_str} for station {station_name}")
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
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.ehlo()
            
            if self.use_auth and self.smtp_username and self.smtp_password:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
            
            server.quit()
            logger.info("✅ Email service connection test successful")
            return True
        except Exception as e:
            logger.error(f"❌ Email service connection test failed: {str(e)}")
            return False

# Global email service instance
email_service = EmailService()

def send_water_level_alert(user_email, station_name, station_id, 
                          current_prediction, min_level, max_level, threshold_percentage=0.9):
    """Convenience function to send water level alert."""
    return email_service.send_water_level_alert(
        user_email, station_name, station_id, 
        current_prediction, min_level, max_level, threshold_percentage
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
        print(f"📧 SMTP Server: {email_service.smtp_server}:{email_service.smtp_port}")
        print(f"📧 From: {email_service.from_name} <{email_service.from_email}>")
        print(f"📧 Default receivers: {', '.join(email_service.default_receivers)}")
        
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
        print("\nCurrent configuration:")
        print(f"- SMTP_SERVER: {email_service.smtp_server}")
        print(f"- SMTP_PORT: {email_service.smtp_port}")
        print(f"- FROM_EMAIL: {email_service.from_email}")
        print(f"- FROM_NAME: {email_service.from_name}")
        print(f"- DEFAULT_RECEIVERS: {', '.join(email_service.default_receivers)}")
