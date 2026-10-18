# AutoStockML — GitHub Actions CI Demo

## Overview
A Flask API serving a simple scikit-learn model.
This project demonstrates Continuous Integration using GitHub Actions.

## DevOps concepts
- Git version control
- GitHub Actions workflows
- Automated dependency installation
- Automated model training
- Automated testing
- CI success and failure reporting

## Setup
Install dependencies:

    python -m pip install -r requirements.txt

Train the model:

    python train_model.py

Run tests:

    python -m pytest -v

Start the application:

    python -m flask --app app run --port 5000

## API endpoints
- GET /
- GET /health
- POST /predict

The model uses synthetic demonstration data and is not
intended for real-world financial forecasting.