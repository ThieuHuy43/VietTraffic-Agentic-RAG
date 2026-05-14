#!/bin/bash
set -e

echo "🚀 Bắt đầu deploy Trợ lý Pháp luật Giao thông..."

if ! command -v docker &> /dev/null
then
    echo "❌ Lỗi: Docker chưa được cài đặt."
    exit 1
fi

echo "📦 Đang Build và Start các services..."
docker compose up -d --build

echo "⏳ Đợi Qdrant và Backend khởi động (10s)..."
sleep 10

echo "📚 Nạp dữ liệu luật vào Qdrant..."
if [ -n "${FORCE_REINGEST:-}" ]; then
    docker compose exec -T -e FORCE_REINGEST="$FORCE_REINGEST" backend_api python ingest.py
else
    docker compose exec -T backend_api python ingest.py
fi

echo "✅ Deploy hoàn tất! Truy cập giao diện tại http://localhost:8000"
