# Fast-Api-inventory-tracker
# to run main .py (uvicorn main:app --reload)# daily-inventory
# Generate & save last 30 days, daily
curl -X POST http://127.0.0.1:8000/profits/generate \
  -H 'Content-Type: application/json' -H 'X-API-Key: @uzair143' \
  -d '{"start_date":"2025-09-27","end_date":"2025-10-27","granularity":"daily"}'

# Read profits
curl 'http://127.0.0.1:8000/profits?granularity=daily&start_date=2025-09-27&end_date=2025-10-27' \
  -H 'X-API-Key: @uzair143'
