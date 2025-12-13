#!/bin/bash

# =============================================================================
# SYNTHETA ENVIRONMENT CONFIGURATION SCRIPT
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}==============================================================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}==============================================================================${NC}"
}

# Default to development if no argument provided
ENVIRONMENT=${1:-development}

print_header "SYNTHETA ENVIRONMENT SETUP"

case $ENVIRONMENT in
    "development"|"dev")
        print_status "Setting up DEVELOPMENT environment..."
        
        # Copy global development environment
        if [ -f .env.development ]; then
            cp .env.development .env
            print_status "Global development environment configured"
        else
            print_error "Global .env.development file not found!"
            exit 1
        fi
        
        # Frontend development environment
        if [ -f apps/frontend/.env.local ]; then
            print_status "Frontend already configured for development"
        else
            print_warning "Frontend .env.local not found, using example"
            if [ -f apps/frontend/.env.example ]; then
                cp apps/frontend/.env.example apps/frontend/.env.local
            fi
        fi
        
        # Backend development environment
        if [ -f apps/backend/.env ]; then
            print_status "Backend development environment exists"
        else
            print_warning "Backend .env not found, creating from example"
            if [ -f apps/backend/.env.example ]; then
                cp apps/backend/.env.example apps/backend/.env
            fi
        fi
        
        # Engine development environment  
        if [ -f apps/engine/.env ]; then
            print_status "Engine development environment exists"
        else
            print_warning "Engine .env not found, creating from example"
            if [ -f apps/engine/.env.example ]; then
                cp apps/engine/.env.example apps/engine/.env
            fi
        fi
        
        print_status "Development environment setup complete!"
        print_status "You can now run: npm run dev (frontend) or uvicorn main:app --reload (backend)"
        ;;
        
    "production"|"prod")
        print_status "Setting up PRODUCTION environment..."
        
        # Copy global production environment
        if [ -f .env.production ]; then
            cp .env.production .env
            print_status "Global production environment configured"
        else
            print_error "Global .env.production file not found!"
            exit 1
        fi
        
        # Frontend production environment
        if [ -f apps/frontend/.env.production ]; then
            cp apps/frontend/.env.production apps/frontend/.env.local
            print_status "Frontend production environment configured"
        else
            print_error "Frontend .env.production not found!"
            exit 1
        fi
        
        # Backend production environment
        if [ -f apps/backend/.env.railway ]; then
            cp apps/backend/.env.railway apps/backend/.env
            print_status "Backend production environment configured"
        else
            print_warning "Backend .env.railway not found"
        fi
        
        # Engine production environment
        if [ -f apps/engine/.env.railway ]; then
            cp apps/engine/.env.railway apps/engine/.env
            print_status "Engine production environment configured"
        else
            print_warning "Engine .env.railway not found"
        fi
        
        print_status "Production environment setup complete!"
        print_status "Environment variables are configured for Railway deployment"
        ;;
        
    *)
        print_error "Invalid environment: $ENVIRONMENT"
        print_status "Usage: ./scripts/set-environment.sh [development|production]"
        print_status "  development (default) - Setup for local development"
        print_status "  production           - Setup for Railway deployment"
        exit 1
        ;;
esac

print_header "ENVIRONMENT VERIFICATION"

# Verify environment setup
if [ -f .env ]; then
    ENV_TYPE=$(grep "ENVIRONMENT=" .env | cut -d'=' -f2)
    print_status "Global environment: $ENV_TYPE"
else
    print_error "Global .env file not found!"
fi

if [ -f apps/frontend/.env.local ]; then
    FRONTEND_BACKEND_URL=$(grep "NEXT_PUBLIC_BACKEND_URL=" apps/frontend/.env.local | cut -d'=' -f2)
    print_status "Frontend backend URL: $FRONTEND_BACKEND_URL"
else
    print_warning "Frontend .env.local not found"
fi

if [ -f apps/backend/.env ]; then
    BACKEND_HOST=$(grep "HOST=" apps/backend/.env | cut -d'=' -f2 2>/dev/null || echo "Not specified")
    print_status "Backend host: $BACKEND_HOST"
else
    print_warning "Backend .env not found"
fi

print_header "NEXT STEPS"

case $ENVIRONMENT in
    "development"|"dev")
        echo -e "${GREEN}Development Environment Ready!${NC}"
        echo ""
        echo "To start the development stack:"
        echo "  1. Start the database: docker-compose up -d postgres redis"
        echo "  2. Start the backend: cd apps/backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000"
        echo "  3. Start the frontend: cd apps/frontend && npm run dev"
        echo "  4. Start the engine: cd apps/engine && uvicorn main:app --reload --host 0.0.0.0 --port 8001"
        echo ""
        echo "URLs:"
        echo "  Frontend: http://localhost:3000"
        echo "  Backend:  http://localhost:8000"
        echo "  Engine:   http://localhost:8001"
        ;;
    "production"|"prod")
        echo -e "${GREEN}Production Environment Ready!${NC}"
        echo ""
        echo "To deploy to Railway:"
        echo "  1. Commit your changes: git add . && git commit -m 'production config'"
        echo "  2. Push to main branch: git push origin main"
        echo "  3. Railway will auto-deploy your services"
        echo ""
        echo "Production URLs:"
        echo "  Frontend: https://www.syntheta.in"
        echo "  Backend:  https://beneficial-gratitude-production-4aef.up.railway.app"
        ;;
esac

echo ""
print_status "Environment setup completed successfully!"