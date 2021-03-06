## 🛒 Категоризация названий товаров

### 📚 Данные для решения задачи

```
demo/data
├── categories.csv
└── products.csv
```

**categories.csv** - файл с категориями на маркетплейсе. У каждой категории есть id, заголовок и путь в дереве
категорий.

**products.csv** - файл с товарами на маркетплейсе. У каждого товара есть id, заголовок и id категории, которой он
принадлежит. Товар всегда принадлежит листовой категории в дереве.

### ⭐ Результаты

* `model_svc.joblib` - Модель, которая умеет предсказывать title -> category
* `vectorizer.joblib`- Векторайзер
* `demo.py` + `demo_requirements.md` - Скрипт и его зависимости с примером работы модели. Live Demo с использованием `input()`
* `kazanexpress_report.ipynb` - Отчет о проделанной работе