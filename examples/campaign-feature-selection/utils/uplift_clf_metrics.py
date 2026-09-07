import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.style.use('ggplot')

from sklearn.utils import resample
from sklift.metrics import qini_curve, qini_auc_score, uplift_at_k, uplift_curve
from tqdm import tqdm

from scipy.stats import ttest_ind


def fact_predict(target, uplift, treatment, n_bins=10):
    # запишем результаты (таргет, аплифт и флаг ЦГ)
    df = pd.DataFrame({'target': target, 'treatment': treatment, 'uplift': uplift})
    df = df.sort_values('uplift', ascending=False)

    # разобьем на равномощные бины по аплифту
    df['bin'] = (n_bins - 1) - pd.qcut(df['uplift'], q=n_bins, labels=False, duplicates='drop')

    # посчитаем фактический алифт в бине (разницу между целевой и контрольной группами по таргету)
    substract = lambda x: x.iloc[1] - x.iloc[0]

    fact_predict = df.groupby('bin', as_index=False)['uplift'].mean()  # MEAN!
    fact_predict['target'] = \
        df.groupby(['bin', 'treatment'], as_index=False).agg({'target': 'mean'}).groupby('bin', as_index=False)[
            'target'].apply(substract)['target']  # MEAN!

    return fact_predict


def check_stat_sig_first_bin(target, uplift, treatment, n_bins=10):
    df = pd.DataFrame({'target': target, 'treatment': treatment, 'uplift': uplift})
    df = df.sort_values('uplift', ascending=False)

    # разобьем на равномощные бины по аплифту
    df['bin'] = (n_bins - 1) - pd.qcut(df['uplift'], q=n_bins, labels=False, duplicates='drop')
    
    treat = df[(df['bin'] == 0) & (df['treatment'] == 1)]['target'].values
    ctrl = df[(df['bin'] == 0) & (df['treatment'] == 0)]['target'].values

    return ttest_ind(treat, ctrl)


def plot_uplift_curve(y_true, uplift, treatment, ax=None):
    x_actual, y_actual = uplift_curve(y_true, uplift, treatment)
    x_random, y_random = x_actual, x_actual * y_actual[-1] / len(y_true)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(x_actual, y_actual, label='model')
    ax.plot(x_random, y_random, label='random')
    ax.legend(loc='upper right')
    ax.set_ylabel('Gain')
    ax.set_xlabel('Population')
    ax.set_title('Uplift Curve')
    

def evaluate_uplift_with_bootstrap(y_true, uplift_score, treatment, 
                                    k_cuts=[0.05, 0.10, 0.30], 
                                    n_bootstrap=500,
                                    save=False,
                                    save_path='./img',
                                    save_prefix=''
                                   ):
    """
    Считает AUQC и Uplift@k с доверительными интервалами.
    k_cuts: список долей для проверки 
    """
    if save and not os.path.exists(save_path):
        print(f'Creating path {save_path} for img saving')
        os.makedirs(save_path, exist_ok=True)
    
    # Подготовка данных
    df_orig = pd.DataFrame({'y': y_true, 'score': uplift_score, 'w': treatment})
    
    metrics_list = []
    
    # --- Вспомогательная функция расчета метрик для одного батча ---
    def calculate_metrics(df_sub):
        # Сортируем по скору
        df_sub = df_sub.sort_values('score', ascending=False).reset_index(drop=True)
        y_true, uplift, treatment = df_sub['y'].values, df_sub['score'].values, df_sub['w'].values

        _, qini_curve_ = qini_curve(y_true, uplift, treatment)
        auqc = qini_auc_score(y_true, uplift, treatment)
        
        res = {'AUQC': auqc}
        
        # 2. Расчет Uplift@k
        for k in k_cuts:
            res[f'Uplift@{int(k*100)}%'] = uplift_at_k(y_true, uplift, treatment, strategy='by_group', k=k)

        return res, qini_curve_

    # --- Запуск Bootstrap ---
    qini_curves = []
    
    print(f"Запуск Bootstrap ({n_bootstrap} итераций)...")
    for i in tqdm(range(n_bootstrap)):
        # Ресемплинг
        boot_df = resample(df_orig, replace=True, random_state=i)
        
        met, curve = calculate_metrics(boot_df)
        metrics_list.append(met)
        
        # Для графика нормализуем кривую по длине (интерполируем к 100 точкам)
        x_norm = np.linspace(0, 100, len(curve))
        qini_interp = np.interp(np.linspace(0, 100, 100), x_norm, curve)
        qini_curves.append(qini_interp)

    # --- Агрегация результатов ---
    metrics_df = pd.DataFrame(metrics_list)
    summary = metrics_df.describe(percentiles=[0.05, 0.5, 0.95]).T
    summary = summary[['5%', 'mean', '95%']]
    summary.columns = ['CI_Lower_5%', 'Mean_Value', 'CI_Upper_95%']
    
    # --- Визуализация ---
    fig = plt.figure(figsize=(10, 6))
    qini_arr = np.array(qini_curves)
    
    mean_curve = np.mean(qini_arr, axis=0)
    lower_curve = np.percentile(qini_arr, 5, axis=0)
    upper_curve = np.percentile(qini_arr, 95, axis=0)
    
    x_axis = np.linspace(0, 100, 100)
    plt.plot(x_axis, mean_curve, label='Mean Qini', color='blue', lw=2)
    plt.fill_between(x_axis, lower_curve, upper_curve, color='blue', alpha=0.15, label='90% CI')
    
    # Random line approximation
    plt.plot([0, 100], [0, mean_curve[-1]], 'r--', label='Random')
    
    plt.title('Bootstrap Qini Curve')
    plt.xlabel('% Population Targeted')
    plt.ylabel('Cumulative Uplift')
    plt.legend()
    plt.grid(alpha=0.3)
    
    if save:
        plt.savefig(f'{save_path}/{save_prefix}_qini_ci.png', dpi=300, bbox_inches='tight')
        
    plt.show()
    
    print("\n=== Uplift Metrics (Bootstrap 90% CI) ===")
    print(summary)
    
    return summary


def show_results(y_true, uplift, treatment, num_bins=20, save=False, save_path='./img', save_prefix=''):
    """Show Uplift Metrics and graphs.
    Args:
        y_true (1d array-like): Correct (true) binary target values.
        uplift (1d array-like): Predicted uplift, as returned by a model.
        treatment (1d array-like): Treatment labels.
        n_bins (int): Number of bins to use in fact-predict graph
        save (bool): If True, then save results as an .png image to save_path
        save_path (str): Path to save results
        save_prefix (str): Prefix to add to name of result image to save
    Returns: pd.DataFrame: Metrics.
    """
    assert isinstance(y_true, np.ndarray) and isinstance(uplift, np.ndarray) and isinstance(treatment,
                                                                                            np.ndarray), 'All arrays must be of class np.ndarray'
    assert len(y_true.shape) == 1 and len(uplift.shape) == 1 and len(
        treatment.shape) == 1, 'All arrays must be of shape (num_samples, ), use .ravel()'
    
    if save and not os.path.exists(save_path):
        print(f'Creating path {save_path} for img saving')
        os.makedirs(save_path, exist_ok=True)
    
    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(8, 21.25), height_ratios=[1, 1, 1, 1, 0.25])

    # uplift curve plot
    u_c = plot_uplift_curve(y_true, uplift, treatment, ax=ax[0])

    # uplift by percentile plot, a.k.a. fact predict plot
    f_p = fact_predict(y_true, uplift, treatment, num_bins)
    sns.barplot(pd.melt(f_p, id_vars=['bin'], value_vars=['uplift', 'target']), x='bin', y='value', hue='variable', ax=ax[1])
    ax[1].set_title('Uplift by percentile with predicted uplift')

    # uplift by percentile plot (without predicted uplift)
    sns.barplot(pd.melt(f_p, id_vars=['bin'], value_vars=['target']), x='bin', y='value', hue='variable', ax=ax[2])
    ax[2].set_title('Uplift by percentile')

    # predicted uplift distribution
    sns.histplot(uplift, bins=150, ax=ax[3])
    ax[3].set_xlabel('Uplift')
    ax[3].set_title(f'Predicted uplift distribution, mean = {np.mean(uplift)}')

    # metrics calculation
    metrics = {}
    metrics['uplift_at_10'] = [uplift_at_k(y_true, uplift, treatment, k=0.1, strategy='by_group')]
    metrics['uplift_at_30'] = [uplift_at_k(y_true, uplift, treatment, k=0.3, strategy='by_group')]

    metrics['qini'] = [qini_auc_score(y_true, uplift, treatment)]
    metrics = pd.DataFrame(metrics)
    
    # metrics plot
    ax[4].grid(False)
    ax[4].set_axis_off()
    ax[4].text(0, 0.8, f"uplift_at_10: {metrics['uplift_at_10'].values[0]}", fontsize=12)
    ax[4].text(0, 0.6, f"uplift_at_30: {metrics['uplift_at_30'].values[0]}", fontsize=12)
    ax[4].text(0, 0.4, f"qini: {metrics['qini'].values[0]}", fontsize=12)
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f'{save_path}/{save_prefix}_uplift_results.png', dpi=300, bbox_inches='tight')
    plt.show();

    return metrics