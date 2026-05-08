Write-Host "Start deploy Viet Traffic Legal Assistant..."

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH."
    exit 1
}

Write-Host "Building and Starting services..."
docker compose up -d --build

Write-Host "Waiting for Qdrant and Backend to start (15s)..."
Start-Sleep -Seconds 15

Write-Host "Generating mock data and Ingesting to Qdrant..."
docker compose exec -T backend_api python utils/mock_data_gen.py
docker compose exec -T backend_api python ingest.py

Write-Host "Deploy completed! Access Frontend at http://localhost:8501"
