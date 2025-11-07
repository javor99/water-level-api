"""Route blueprints."""
from app.routes.auth import auth_bp
from app.routes.municipalities import municipalities_bp
from app.routes.stations import stations_bp
from app.routes.predictions import predictions_bp
from app.routes.subscriptions import subscriptions_bp

__all__ = [
    'auth_bp',
    'municipalities_bp',
    'stations_bp',
    'predictions_bp',
    'subscriptions_bp'
]

