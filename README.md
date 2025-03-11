# test_flowers
search for the most similar plants


Для создания виртуального окружения:
```
python -m venv venv
source venv/bin/activate
```
Необходимо установить зависимости:
```
pip install -r requirements.txt
```

## Сборка докер образа:
```
docker build -t plant-search-api .
```

## Запуск докер контейнера:
```
docker run -p 5000:5000 plant-search-api
```

Запуск происходит на порту 5000. API доступен на http://localhost:5000/search.
