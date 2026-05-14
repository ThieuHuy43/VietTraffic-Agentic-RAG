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

echo "📚 Sinh dữ liệu Mock và nạp (Ingest) vào Qdrant..."
docker compose exec -T backend_api python utils/mock_data_gen.py
docker compose exec -T backend_api python ingest.py

echo "✅ Deploy hoàn tất! Truy cập giao diện tại http://localhost:8000"
