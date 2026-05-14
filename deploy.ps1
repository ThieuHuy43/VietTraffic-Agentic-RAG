Write-Host "Start deploy Viet Traffic Legal Assistant..."

if (-not (Get-Command "docker" -ErrorAction SilentlyContinue)) {
    Write-Error "Docker is not installed or not in PATH."
    exit 1
}

Write-Host "Building and Starting services..."
docker compose up -d --build

Write-Host "Waiting for Qdrant and Backend to start (15s)..."
Start-Sleep -Seconds 15

Write-Host "Ingesting legal data to Qdrant..."
if ($env:FORCE_REINGEST) {
    docker compose exec -T -e FORCE_REINGEST=$env:FORCE_REINGEST backend_api python ingest.py
}
else {
    docker compose exec -T backend_api python ingest.py
}

Write-Host "Deploy completed! Access Web UI at http://localhost:8000" -ForegroundColor Green
