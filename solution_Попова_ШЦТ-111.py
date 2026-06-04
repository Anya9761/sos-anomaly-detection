import os
import sys
import glob
import time
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

warnings.filterwarnings('ignore')

# ────────────────────── Настройки ────────────────────────
SCORE_THRESHOLD = 10.0  # порог robust z-score 
MIN_OBS_PER_BRAND = 10  # минимум наблюдений для per-brand статистики
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

# Автоматический поиск папки с данными
_BASE = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.join(_BASE, 'data_train'),
    os.path.join(_BASE, '..', 'data_train'),
    os.path.join(os.path.expanduser('~'), 'Desktop', 'data_train'),
]:
    if os.path.isdir(_candidate):
        DATA_PATH = _candidate
        break
else:
    DATA_PATH = os.path.join(_BASE, 'data_train')

# ──────────────────────────────────────────────────────────


def load_data(data_path: str) -> pd.DataFrame:
    """Загружает все parquet-файлы из data_train."""
    files = sorted(glob.glob(os.path.join(data_path, 'month=*', '*.parquet')))
    if not files:
        raise FileNotFoundError(f"Parquet-файлы не найдены в {data_path}")

    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    df['Weight'] = pd.to_numeric(df['Weight'], errors='coerce')
    df['researchdate'] = df['researchdate'].astype(str)
    # BrandID как строка
    df['BrandID'] = df['BrandID'].astype(str)

    return df


def compute_daily_ots(df: pd.DataFrame) -> pd.DataFrame:
    """
    Вычисляет daily_ots(i, j, k) = Weight(i, k) * count_rows(i, j, k).

    count_rows = число строк с одинаковым (SubjectID, BrandID, CategoryNameDelivery,
    researchdate) среди строк с BrandinDelivery = 1.

    Возвращает DataFrame со столбцами:
        SubjectID, BrandID, CategoryNameDelivery, researchdate,
        Brand, Weight, count_rows, daily_ots
    """
    # Только строки, входящие в поставку SoS
    mask = (
        (df['BrandinDelivery'] == 1.0)
        & (df['CategoryNameDelivery'].notna())
    )
    df_valid = df[mask].copy()

    grp_cols = ['SubjectID', 'BrandID', 'CategoryNameDelivery', 'researchdate']
    agg = df_valid.groupby(grp_cols, sort=False).agg(
        count_rows=('BrandinDelivery', 'size'),
        Weight=('Weight', 'first'),
        Brand=('Brand', 'first'),
    ).reset_index()

    agg['daily_ots'] = agg['Weight'] * agg['count_rows']
    return agg


def _robust_mad(vals: np.ndarray):
    """
    Возвращает (median, mad, all_equal) для массива vals.

    Трёхуровневая защита от MAD == 0:
      1. MAD из данных
      2. std / 1.4826  — если MAD == 0, но разброс есть
      3. all_equal=True — если все значения одинаковы (std тоже 0),
         вызывающий код должен пропустить этот бренд (аномалий нет)
    """
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))

    if mad > 1e-9:
        return med, mad, False

    # MAD == 0: больше половины значений совпадают с медианой
    std_val = float(np.std(vals, ddof=0))
    if std_val > 1e-9:
        return med, std_val / 1.4826, False

    # Все значения одинаковы — аномалий нет
    return med, 1.0, True


def detect_anomalies(
    daily_ots_df: pd.DataFrame,
    score_threshold: float = SCORE_THRESHOLD,
    min_obs: int = MIN_OBS_PER_BRAND,
) -> pd.DataFrame:
    """
    Для каждой (BrandID, CategoryDelivery) вычисляет Robust Z-Score
    и отмечает аномально высокие daily_ots.

    Основной сигнал — score_ots (z-score по daily_ots).
    Дополнительно вычисляется score_count (z-score по count_rows) —
    используется только в поле reason для диагностики.
    combined_score = max(score_ots, 0.5 * score_count) сохраняется
    в колонку score, но пороговым фильтром является score_ots
    (через предфильтр daily_ots > thresh_ots).

    Возвращает DataFrame с причинами аномалий:
        SubjectID, researchdate, BrandID, Brand, CategoryDelivery,
        daily_ots, score, threshold, reason
    """
    all_ots = daily_ots_df['daily_ots'].values.astype(float)

    # Глобальные параметры (запасной вариант для редких брендов)
    g_med, g_mad, g_all_equal = _robust_mad(all_ots)
    g_thresh_ots = g_med + (score_threshold / 0.6745) * g_mad

    rows = []

    for (brand_id, cat), grp in daily_ots_df.groupby(
        ['BrandID', 'CategoryNameDelivery'], sort=False
    ):
        vals = grp['daily_ots'].values.astype(float)
        count_vals = grp['count_rows'].values.astype(float)
        n = len(vals)

        # MAD для count_rows (диагностика) — тоже через _robust_mad
        count_med, count_mad, _ = _robust_mad(count_vals)

        if n >= min_obs:
            med, mad, all_equal = _robust_mad(vals)
            if all_equal:
                # Все значения одинаковы — аномалий нет
                continue
            thresh_ots = med + (score_threshold / 0.6745) * mad
        else:
            med, mad, thresh_ots = g_med, g_mad, g_thresh_ots

        outlier_rows = grp[grp['daily_ots'] > thresh_ots]

        for _, r in outlier_rows.iterrows():
            score_ots = 0.6745 * (r['daily_ots'] - med) / mad
            score_count = (
                0.6745 *
                (r['count_rows'] - count_med) /
                count_mad
            )

            combined_score = max(score_ots, 0.5 * score_count)
            rows.append({
                'SubjectID': r['SubjectID'],
                'researchdate': r['researchdate'],
                'BrandID': r['BrandID'],
                'Brand': r['Brand'],
                'CategoryDelivery': r['CategoryNameDelivery'],
                'daily_ots': round(float(r['daily_ots']), 3),
                'score': round(float(combined_score), 4),
                'threshold': score_threshold,
                'reason': (
                    f'CombinedScore={combined_score:.2f}; '
                    f'OTS_Z={score_ots:.2f}; '
                    f'COUNT_Z={score_count:.2f}; '
                    f'daily_ots={r["daily_ots"]:.1f} > thresh={thresh_ots:.1f} '
                    f'(median={med:.1f}, MAD={mad:.1f}, n={n})'
                ),
            })

    return pd.DataFrame(rows)


def save_outputs(anomaly_reasons: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """Сохраняет anomalies.csv и anomaly_reasons.csv."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    anomalies = (
        anomaly_reasons[['SubjectID', 'researchdate']]
        .drop_duplicates()
        .sort_values(['researchdate', 'SubjectID'])
        .reset_index(drop=True)
    )
    anomalies.to_csv(os.path.join(output_dir, 'anomalies.csv'), index=False)

    cols_out = ['SubjectID', 'researchdate', 'BrandID', 'Brand',
                'CategoryDelivery', 'daily_ots', 'score', 'threshold', 'reason']
    anomaly_reasons[cols_out].sort_values(
        ['researchdate', 'SubjectID', 'BrandID']
    ).to_csv(os.path.join(output_dir, 'anomaly_reasons.csv'), index=False)

    return anomalies


def _build_anomaly_set(anomalies: pd.DataFrame) -> set:
    return set(
        zip(anomalies['SubjectID'].astype(str), anomalies['researchdate'].astype(str))
    )


def _filter_clean(df_valid, anomaly_set):
    keys = pd.Series(
        list(
            zip(
                df_valid['SubjectID'].astype(str),
                df_valid['researchdate'].astype(str),
            )
        ),
        index=df_valid.index,
    )
    return df_valid[~keys.isin(anomaly_set)]


def _ots_from_daily(daily_ots_df: pd.DataFrame, anomaly_set: set, group_col: str):
    """
    Считает суммарный OTS (сумма daily_ots = Weight * count_rows)
    по group_col до и после удаления аномалий.
    Использует агрегированный daily_ots_df, а не сырой df.
    """
    before = daily_ots_df.groupby(group_col)['daily_ots'].sum()

    if anomaly_set:
        keys = pd.Series(
            list(zip(
                daily_ots_df['SubjectID'].astype(str),
                daily_ots_df['researchdate'].astype(str),
            )),
            index=daily_ots_df.index,
        )
        clean = daily_ots_df[~keys.isin(anomaly_set)]
    else:
        clean = daily_ots_df

    after = clean.groupby(group_col)['daily_ots'].sum().reindex(before.index).fillna(0)
    return before, after


def create_plots(
    df: pd.DataFrame,
    anomaly_reasons: pd.DataFrame,
    anomalies: pd.DataFrame,
    output_dir: str,
    daily_ots_df: pd.DataFrame = None,
) -> None:
    plots_dir = os.path.join(output_dir, 'plots')
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    anomaly_set = _build_anomaly_set(anomalies) if len(anomalies) > 0 else set()

    # ── 1. Общий OTS по дням до / после ──────────────────────
    if daily_ots_df is not None:
        ots_before_s, ots_after_s = _ots_from_daily(daily_ots_df, anomaly_set, 'researchdate')
    else:
        df_valid = df[df['BrandinDelivery'] == 1.0].copy()
        df_valid['Weight'] = pd.to_numeric(df_valid['Weight'], errors='coerce')
        df_clean = _filter_clean(df_valid, anomaly_set) if anomaly_set else df_valid
        ots_before_s = df_valid.groupby('researchdate')['Weight'].sum().sort_index()
        ots_after_s = df_clean.groupby('researchdate')['Weight'].sum().reindex(ots_before_s.index).fillna(0)

    ots_before_s = ots_before_s.sort_index()
    ots_after_s = ots_after_s.reindex(ots_before_s.index).fillna(0)
    ots_df = pd.DataFrame({'ots_before': ots_before_s, 'ots_after': ots_after_s}).reset_index()
    ots_df.columns = ['researchdate', 'ots_before', 'ots_after']

    avg_b = ots_df['ots_before'].mean()
    avg_a = ots_df['ots_after'].mean()
    pct = (avg_a - avg_b) / avg_b * 100 if avg_b != 0 else 0.0

    fig, ax = plt.subplots(figsize=(14, 5))
    n = len(ots_df)
    xs = range(n)
    ax.plot(xs, ots_df['ots_before'], label='OTS before', color='green', linewidth=1.2)
    ax.plot(xs, ots_df['ots_after'], label='OTS after', color='red', linewidth=1.2)
    step = max(1, n // 25)
    ax.set_xticks(list(range(0, n, step)))
    ax.set_xticklabels(ots_df['researchdate'].iloc[::step], rotation=45, ha='right', fontsize=7)
    ax.set_title(
        f'Изменение общего OTS до и после удаления аномалий\n'
        f'avg_before={avg_b:.0f}, avg_after={avg_a:.0f}, Δ={pct:.2f}%'
    )
    ax.set_xlabel('Дата')
    ax.set_ylabel('Суммарный OTS (Weight × count_rows)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'total_ots_before_after.png'), dpi=120)
    plt.close()

    # ── 2. Изменение OTS по CategoryDelivery (%) ─────────────
    if daily_ots_df is not None:
        cat_before, cat_after = _ots_from_daily(daily_ots_df, anomaly_set, 'CategoryNameDelivery')
    else:
        df_valid = df[df['BrandinDelivery'] == 1.0].copy()
        df_valid['Weight'] = pd.to_numeric(df_valid['Weight'], errors='coerce')
        df_clean = _filter_clean(df_valid, anomaly_set) if anomaly_set else df_valid
        cat_before = df_valid.groupby('CategoryNameDelivery')['Weight'].sum()
        cat_after = df_clean.groupby('CategoryNameDelivery')['Weight'].sum().reindex(cat_before.index).fillna(0)

    cat_pct = ((cat_after - cat_before) / cat_before * 100).fillna(0).sort_values()

    colors = ['#d62728' if v < 0 else '#1f77b4' for v in cat_pct.values]
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(range(len(cat_pct)), cat_pct.values, color=colors)
    ax.set_xticks(range(len(cat_pct)))
    ax.set_xticklabels(cat_pct.index, rotation=45, ha='right', fontsize=7)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_title('Изменение суммарного OTS по категориям (%) после очистки')
    ax.set_xlabel('Категория')
    ax.set_ylabel('Изменение OTS (%)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'category_ots_change.png'), dpi=120)
    plt.close()

    # ── 3. Количество аномальных респондентов по дням ────────
    if len(anomaly_reasons) > 0:
        daily_count = (
            anomaly_reasons.groupby('researchdate')['SubjectID']
            .nunique()
            .sort_index()
            .reset_index()
        )
        daily_count.columns = ['researchdate', 'count']
        total_pairs = len(anomalies)
        unique_subj = anomalies['SubjectID'].nunique()
    else:
        daily_count = pd.DataFrame({'researchdate': [], 'count': []})
        total_pairs, unique_subj = 0, 0

    fig, ax = plt.subplots(figsize=(14, 5))
    if len(daily_count) > 0:
        m = len(daily_count)
        ax.bar(range(m), daily_count['count'], color='steelblue')
        st = max(1, m // 25)
        ax.set_xticks(list(range(0, m, st)))
        ax.set_xticklabels(daily_count['researchdate'].iloc[::st], rotation=45, ha='right', fontsize=7)
    ax.set_title(
        f'Количество аномальных респондентов по дням\n'
        f'Всего пар (SubjectID, date): {total_pairs}, '
        f'уникальных SubjectID: {unique_subj}'
    )
    ax.set_xlabel('Дата')
    ax.set_ylabel('Кол-во аномальных респондентов')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'daily_anomaly_count.png'), dpi=120)
    plt.close()


# ═══════════════ Аналитические функции ═══════════════════

def plot_ots_by_feature(df, anomalies, feature_col, title=None, save_path=None,
                        daily_ots_df=None):
    """График OTS до/после по любому срезу (пол, возраст, регион и т.д.).
    Если передан daily_ots_df — OTS считается корректно (Weight * count_rows).
    Для признаков только в сыром df (демография) делается join через SubjectID.
    """
    anomaly_set = _build_anomaly_set(anomalies)

    if daily_ots_df is not None and feature_col in daily_ots_df.columns:
        before, after = _ots_from_daily(daily_ots_df, anomaly_set, feature_col)
    else:
        dv = df[df['BrandinDelivery'] == 1.0].copy()
        dv['Weight'] = pd.to_numeric(dv['Weight'], errors='coerce')
        if daily_ots_df is not None and feature_col in dv.columns:
            feat_map = dv.dropna(subset=[feature_col]).groupby('SubjectID')[feature_col].first()
            merged = daily_ots_df.copy()
            merged[feature_col] = merged['SubjectID'].map(feat_map)
            merged = merged.dropna(subset=[feature_col])
            before, after = _ots_from_daily(merged, anomaly_set, feature_col)
        else:
            before = dv.groupby(feature_col)['Weight'].sum().sort_index()
            after = _filter_clean(dv, anomaly_set).groupby(feature_col)['Weight'].sum()
            after = after.reindex(before.index).fillna(0)

    xs = range(len(before))
    fig, ax = plt.subplots(figsize=(max(10, len(before) * 0.5), 5))
    ax.bar([i - 0.2 for i in xs], before.values, width=0.38, label='До', color='#1f77b4', alpha=0.8)
    ax.bar([i + 0.2 for i in xs], after.values, width=0.38, label='После', color='#d62728', alpha=0.8)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(before.index, rotation=45, ha='right', fontsize=9)
    ax.set_title(title or f'OTS до/после по признаку «{feature_col}»')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


def plot_ots_by_resource(df, anomalies, resource_col='ResourceName', save_path=None,
                         daily_ots_df=None):
    """График OTS до/после по характеристикам ресурсов."""
    plot_ots_by_feature(df, anomalies, resource_col,
                        title=f'OTS до/после по {resource_col}', save_path=save_path,
                        daily_ots_df=daily_ots_df)


def plot_ots_by_category_level(df, anomalies, cat_col='Category1', save_path=None,
                               daily_ots_df=None):
    """График OTS до/после по уровням категорий (Category1/2/3/CategoryNameDelivery)."""
    plot_ots_by_feature(df, anomalies, cat_col,
                        title=f'OTS до/после по {cat_col}', save_path=save_path,
                        daily_ots_df=daily_ots_df)


def get_anomalous_queries(df, subject_id, researchdate) -> pd.DataFrame:
    """Таблица поисковых запросов аномального респондента за конкретный день."""
    mask = (df['SubjectID'] == subject_id) & (df['researchdate'].astype(str) == str(researchdate))
    cols = ['QueryText', 'Brand', 'CategoryNameDelivery', 'BrandinDelivery', 'Weight', 'ResourceName']
    return df[mask][cols].reset_index(drop=True)


def plot_brand_ots_over_time(df, brand_id, anomalies, save_path=None):
    """График дневного OTS конкретного бренда до/после очистки."""
    dv = df[(df['BrandinDelivery'] == 1.0) & (df['BrandID'].astype(str) == str(brand_id))].copy()
    dv['Weight'] = pd.to_numeric(dv['Weight'], errors='coerce')

    before = dv.groupby('researchdate')['Weight'].sum().sort_index()
    anomaly_set = _build_anomaly_set(anomalies)
    after = _filter_clean(dv, anomaly_set).groupby('researchdate')['Weight'].sum()
    after = after.reindex(before.index).fillna(0)

    brand_name = dv['Brand'].iloc[0] if len(dv) > 0 else str(brand_id)
    n = len(before)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(range(n), before.values, label='До', color='green', linewidth=1.3)
    ax.plot(range(n), after.values, label='После', color='red', linewidth=1.3)
    step = max(1, n // 20)
    ax.set_xticks(list(range(0, n, step)))
    ax.set_xticklabels(before.index[::step], rotation=45, ha='right', fontsize=7)
    ax.set_title(f'OTS бренда «{brand_name}» (ID={brand_id}) по дням')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.close()


# ═══════════════ Основная функция ════════════════════════

def main():
    t0 = time.time()

    print("=" * 60)
    print("Поиск аномальных респондентов в SoS (Robust Z-Score per Brand)")
    print("=" * 60)

    # 1. Загрузка данных
    print(f"\n[1/5] Загрузка данных из: {DATA_PATH}")
    df = load_data(DATA_PATH)
    n_total = len(df)
    n_subj = df['SubjectID'].nunique()
    dates = sorted(df['researchdate'].unique())
    print(f"      Записей: {n_total:,} | Респондентов: {n_subj:,} | Дат: {len(dates)}")
    print(f"      Период: {dates[0]} — {dates[-1]}")

    # 2. Вычисление daily_ots
    print("\n[2/5] Вычисление daily_ots...")
    daily_ots = compute_daily_ots(df)
    print(f"      Уникальных (Subject, Brand, Category, Date): {len(daily_ots):,}")
    print(f"      daily_ots: median={daily_ots['daily_ots'].median():.1f}, "
          f"max={daily_ots['daily_ots'].max():.1f}")

    # 3. Обнаружение аномалий
    print(f"\n[3/5] Обнаружение аномалий (threshold={SCORE_THRESHOLD}, min_obs={MIN_OBS_PER_BRAND})...")
    anomaly_reasons = detect_anomalies(
        daily_ots, score_threshold=SCORE_THRESHOLD, min_obs=MIN_OBS_PER_BRAND
    )
    n_triggers = len(anomaly_reasons)
    n_pairs = len(anomaly_reasons[['SubjectID', 'researchdate']].drop_duplicates())
    n_uniq_s = anomaly_reasons['SubjectID'].nunique() if n_triggers > 0 else 0
    print(f"      Триггеров (Subject, Brand, Category, Date): {n_triggers:,}")
    print(f"      Уникальных пар (Subject, Date):  {n_pairs:,}")
    print(f"      Уникальных SubjectID: {n_uniq_s:,}")
    if n_triggers > 0:
        print(f"      Score range: {anomaly_reasons['score'].min():.2f} — {anomaly_reasons['score'].max():.2f}")

    # 4. Сохранение результатов
    print(f"\n[4/5] Сохранение файлов в {OUTPUT_DIR}/...")
    anomalies = save_outputs(anomaly_reasons, OUTPUT_DIR)
    print(f"      anomalies.csv:       {len(anomalies):,} строк")
    print(f"      anomaly_reasons.csv: {n_triggers:,} строк")

    # 5. Графики
    print("\n[5/5] Построение графиков...")
    create_plots(df, anomaly_reasons, anomalies, OUTPUT_DIR, daily_ots_df=daily_ots)
    print("      Графики сохранены в output/plots/")

    # ── Метрики качества ──
    # OTS считается корректно: сумма daily_ots = Weight * count_rows
    total_ots_before = daily_ots['daily_ots'].sum()

    if len(anomalies) > 0:
        aset = _build_anomaly_set(anomalies)
        keys = pd.Series(
            list(zip(daily_ots['SubjectID'].astype(str), daily_ots['researchdate'].astype(str))),
            index=daily_ots.index,
        )
        daily_ots_clean = daily_ots[~keys.isin(aset)]
    else:
        daily_ots_clean = daily_ots
    total_ots_after = daily_ots_clean['daily_ots'].sum()
    ots_retention = total_ots_after / total_ots_before * 100 if total_ots_before > 0 else 100.0

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("ИТОГ")
    print("=" * 60)
    print(f"  Время выполнения:        {elapsed:.1f} сек")
    print(f"  Удалено пар (subj/date): {n_pairs:,} из {n_subj * len(dates):,} возможных")
    pct_removed = n_uniq_s / n_subj * 100 if n_subj > 0 else 0
    print(f"  Доля удалённых субъектов:{pct_removed:.2f}%")
    print(f"  OTS retained:            {ots_retention:.2f}%")
    print(f"  Алгоритм:                Robust Z-Score per Brand (MAD-based)")
    print(f"  Порог:                   {SCORE_THRESHOLD}")
    print(f"\n  Файлы:")
    print(f"    {OUTPUT_DIR}/anomalies.csv")
    print(f"    {OUTPUT_DIR}/anomaly_reasons.csv")
    print(f"    {OUTPUT_DIR}/plots/total_ots_before_after.png")
    print(f"    {OUTPUT_DIR}/plots/category_ots_change.png")
    print(f"    {OUTPUT_DIR}/plots/daily_anomaly_count.png")

    return df, daily_ots, anomaly_reasons, anomalies


if __name__ == '__main__':
    df, daily_ots, anomaly_reasons, anomalies = main()
