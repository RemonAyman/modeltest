from vercel_wsgi import make_handler
from app import app as application

# Vercel Python serverless expects a callable named `handler`.
handler = make_handler(application)
