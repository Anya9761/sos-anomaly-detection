# Поиск аномальных респондентов SoS

## Запуск

Перед запуском скрипта (если не установлен): 
    pip install -r requirements.txt

```bash
python solution_Попова_ШЦТ-111.py
```

Создаёт папку `output/` с обязательными файлами.

Скрипт автоматически ищет папку data_train рядом с решением,
на уровень выше или на рабочем столе пользователя.

## Алгоритм

**Метрика:** `daily_ots(i,j,k) = Weight(i,k) × count_rows(i,j,k)`

**Метод:** Robust Z-Score (MAD) на уровне каждого бренда

Для каждой пары `(BrandID, CategoryNameDelivery)`:
1. Собираем все значения `daily_ots(i,j,k)` по всем (субъект, день)
2. Медиана и MAD = median(|x − median|)
3. `score_ots = 0.6745 × (daily_ots − median) / MAD`
4. Аномалия если `score_ots > 10.0` (т.е. `daily_ots > median + 10.0/0.6745 × MAD`)
5. Дополнительно вычисляется `score_count` — z-score по числу строк `count_rows`
6. В выходной файл записывается:
combined_score = max(score_ots, 0.5 × score_count)
Пороговое решение при этом принимается только по score_ots.

**Порог 10.0** — выбран для выделения только экстремальных выбросов.
Robust Z-Score уже адаптируется к масштабу данных через медиану и MAD,
поэтому порог устойчив к изменению абсолютных значений между периодами.
`score_count` используется как диагностический сигнал в поле `reason`.
Для брендов с <10 наблюдений используется глобальный порог (медиана и MAD по всему датасету).

**Защита от MAD == 0** — трёхуровневая:
1. MAD из данных
2. std / 1.4826 — если MAD == 0, но разброс есть
3. Бренд пропускается — если все значения одинаковы (аномалий нет)

**OTS считается корректно:** везде используется `sum(daily_ots) = sum(Weight × count_rows)`,
а не просто сумма весов строк.

## Результаты на данных (Jun–Oct 2025)

| Метрика | Значение |
|----------|----------:|
| Время выполнения | ~4–5 секунд |
| Удалено пар (SubjectID, date) | 2 340 |
| Уникальных SubjectID | 1 204 |
| Доля удалённых субъектов | 5.01% |
| OTS retained | 82.94% |

OTS retained — доля суммарного OTS, сохранившаяся после удаления аномальных наблюдений.

## Выходные файлы

```
output/
├── anomalies.csv              # SubjectID, researchdate (уникальные пары)
├── anomaly_reasons.csv        # SubjectID, researchdate, BrandID, Brand,
│                              # CategoryDelivery, daily_ots, score, threshold, reason
└── plots/
    ├── total_ots_before_after.png
    ├── category_ots_change.png
    └── daily_anomaly_count.png
```

## Аналитические возможности

В коде предусмотрены функции для дополнительного анализа (вызываются вручную после `main()`):

```python
# До/после по характеристикам респондентов
    plot_ots_by_feature(df, anomalies, 'Пол', daily_ots_df=daily_ots,
                        save_path='output/plots/ots_by_pol.png')
    plot_ots_by_feature(df, anomalies, 'Возраст', daily_ots_df=daily_ots,
                        save_path='output/plots/ots_by_vozrast.png')
    plot_ots_by_feature(df, anomalies, 'Регион', daily_ots_df=daily_ots,
                        save_path='output/plots/ots_by_region.png')
    plot_ots_by_feature(df, anomalies, 'Федеральный_округ', daily_ots_df=daily_ots,
                        save_path='output/plots/ots_by_fo.png')

    # До/после по характеристикам ресурсов
    plot_ots_by_resource(df, anomalies, 'ResourceName', daily_ots_df=daily_ots,
                         save_path='output/plots/ots_by_resource.png')
    plot_ots_by_resource(df, anomalies, 'ResourceType', daily_ots_df=daily_ots,
                         save_path='output/plots/ots_by_resource_type.png')
    plot_ots_by_resource(df, anomalies, 'Platform', daily_ots_df=daily_ots,
                         save_path='output/plots/ots_by_platform.png')
    plot_ots_by_resource(df, anomalies, 'UseType', daily_ots_df=daily_ots,
                         save_path='output/plots/ots_by_usetype.png')

    # До/после по уровням категорий
    plot_ots_by_category_level(df, anomalies, 'Category1', daily_ots_df=daily_ots,
                                save_path='output/plots/ots_by_cat1.png')
    plot_ots_by_category_level(df, anomalies, 'Category2', daily_ots_df=daily_ots,
                                save_path='output/plots/ots_by_cat2.png')
    plot_ots_by_category_level(df, anomalies, 'Category3', daily_ots_df=daily_ots,
                                save_path='output/plots/ots_by_cat3.png')

    # Поисковые запросы первого аномального респондента
    if len(anomalies) > 0:
        subj = anomalies['SubjectID'].iloc[0]
        date = anomalies['researchdate'].iloc[0]
        print(f"\nЗапросы аномального респондента {subj} за {date}:")
        print(get_anomalous_queries(df, subject_id=subj, researchdate=date))

    # OTS первого бренда-триггера по дням до/после
    if len(anomaly_reasons) > 0:
        brand = anomaly_reasons['BrandID'].iloc[0]
        plot_brand_ots_over_time(df, brand_id=brand, anomalies=anomalies,
                                 save_path='output/plots/ots_brand_example.png')
```

## Зависимости

```
pip install -r requirements.txt
```
