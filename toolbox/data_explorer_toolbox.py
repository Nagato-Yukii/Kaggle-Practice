"""
一个用于快速进行数据探索性分析(DA)的实用工具模块。
主要功能是提供按目标变量分组的描述性统计信息。
"""
import pandas as pd
from typing import Dict, Any, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from scipy import stats
import numpy as np
from scipy.stats import skew

# 为了在Jupyter Notebook等环境中获得更好的显示效果，我们尝试导入display
# 如果在普通Python环境中运行，它会优雅地降级为使用print
try:
    from IPython.display import display, HTML
except ImportError:
    # 如果没有安装IPython，定义一个备用函数
    def display(x):
        """备用显示函数,使用print。"""
        print(x)
    def HTML(s: str) -> str:
        """备用HTML函数,直接返回字符串。"""
        return s

def check_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    对于陌生的训练数据和测试数据,快速check它们的情况
    1.Shape of Data
    2.df.head(10)
    3.df.info()
    4.df.nunique()
    5.df.describe().transpose()
    6.df missing value check(num,percentage),缺失值热图

    参数:
        df(pd.DataFrame):训练数据
    
    返回:
    """
    #传参数据类型检查
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入 'df' 必须是 pandas DataFrame。")
    
    print("--- Check Data ---")

    print("---------Data Info--------------")
    print("Shape of Train Data && .head() &&info && nunique && describe", df.shape)
    display(df.head(10))
    df.info()
    display(df.nunique())
    display(df.describe().transpose())

    print("----missing value check----")
    missing_train = df.isnull().sum()
    missing_percent_train = (missing_train/len(df))*100
    print(missing_percent_train)

    sns.heatmap(df.isnull(), cbar=False)
    plt.title(f"Missing Values Heatmap Data")
    plt.show()


def check_pair(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    对于陌生的训练数据和测试数据对,分别调用check_data,快速check它们的情况
    参数:
        df_train(pd.DataFrame):训练数据
        df_test(pd.DataFrame):测试数据
    
    返回:
    """
    #传参数据类型检查
    if not isinstance(df_train, pd.DataFrame) or not isinstance(df_test, pd.DataFrame):
        raise TypeError("输入 'df_train' 和 'df_test' 都必须是 pandas DataFrame。")
    
    print("--- Check Data Pair---")

    print("-----------Train------------");check_data(df_train)

    print("-----------Test------------");check_data(df_test)


def class_distribute(df: pd.DataFrame , columnname:str):
    """
    对于dataframe的分类型数据,直接plot它的分布

    参数:
        df(pd.DataFrame):数据名称
        columnname(str):该行数据的名称(必须是分类型数据)
    """

    #传参检查
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入 'df' 必须是 pandas DataFrame。")
    if columnname not in df.columns:
        raise ValueError(f"DataFrame 中缺少名为 '{columnname}' 的列。")
    if (not pd.api.types.is_categorical_dtype(df[columnname].dtype)) and (not pd.api.types.is_object_dtype(df[columnname].dtype)):
        raise TypeError(f"列 '{columnname}' 的数据类型必须是 'category' (分类型)。")
    
    plt.figure(figsize = (12,6))
    stage_fear_counts = df[columnname].value_counts()
    ax = sns.barplot(x=stage_fear_counts.index, y=stage_fear_counts.values, palette="viridis")

    plt.title(f"Distribution of {columnname}", fontsize=15)
    plt.xlabel(columnname)
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    total = len(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.01*total,
                f'{height}\n({height/total:.1%})',  # Count + percentage
                ha='center', va='center', fontsize=10)

    plt.show()


def related_distribute(df: pd.DataFrame , columnname_main:str, columnname:str):
    """
    对于dataframe的两个分类型数据,plot columnname 在columnname_main的分类下的分布

    参数:
        df(pd.DataFrame):数据名称
        columnname_main(str):主导的column的name
        column(str):非主导的column的name
    """
    plt.figure(figsize=(10, 20))
    pd.crosstab(df[columnname_main], df[columnname]).plot(kind='bar', stacked=False, colormap='viridis')
    plt.title(f'{columnname} in {columnname_main}', fontsize=16)
    plt.xlabel(columnname_main, fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title=columnname, bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()


def correlation_analyze(df, columns2analyze=None):
    """
    对DataFrame中的数值列进行系统性的两两相关性分析。

    该函数会为每一对数值列生成一个带有回归线和统计信息的散点图，
    并打印出对相关性强度、方向和显著性的文字解读。

    参数:
    ----------
    df : pandas.DataFrame
        包含待分析数据的DataFrame。

    columns2analyze : list of str, optional (默认=None)
        一个包含待分析列名的列表。
        如果为None,函数会自动选择DataFrame中所有的数值类型列。
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入 'df' 必须是 pandas DataFrame。")
    
    if columns2analyze is None:
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    else:
        numerical_cols = columns2analyze
    
    numerical_cols = [col for col in numerical_cols if not is_id_column(col)]
    if len(numerical_cols) < 2:
        print("错误：至少需要两个数值列才能进行两两分析。")
        return
    print(f"numerical_cols used = {numerical_cols}")
    #生成所有唯一的列组合
    column_pairs = combinations(numerical_cols, 2)

    sns.set_theme(style="whitegrid")

    #循环遍历每一对列，进行分析和绘图
    print(f"--- 开始对 {len(numerical_cols)} 个数值列进行两两相关性分析 ---")
    for col1, col2 in column_pairs:
        plt.figure(figsize=(10, 6))
        # 绘制散点图和回归线
        sns.regplot(x=col1, y=col2, data=df, scatter_kws={'alpha': 0.6})
        # 计算统计数据（移除缺失值以确保计算成功）
        temp_df = df[[col1, col2]].dropna()
        if len(temp_df) < 2:
            print(f"\n跳过 {col1} vs {col2}：移除缺失值后数据不足。")
            continue
        corr_coef, p_value = stats.pearsonr(temp_df[col1], temp_df[col2])
        slope, intercept, _, _, _ = stats.linregress(temp_df[col1], temp_df[col2])
        stats_text = (f"Pearson r = {corr_coef:.2f}\n"
                      f"p-value = {p_value:.4f}\n"
                      f"Regression: y = {slope:.2f}x + {intercept:.2f}")
        plt.gcf().text(0.5, 0.01, stats_text, ha='center', fontsize=10, 
                       bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
        plt.title(f'{col1} vs {col2}', fontsize=14, pad=20)
        plt.xlabel(col1, fontsize=12)
        plt.ylabel(col2, fontsize=12)
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        plt.show()

        # 自动化文字解读
        abs_r = abs(corr_coef)
        # 解读相关强度
        if abs_r >= 0.8: strength = "非常强"
        elif abs_r >= 0.6: strength = "强"
        elif abs_r >= 0.4: strength = "中等"
        elif abs_r >= 0.2: strength = "弱"
        else: strength = "非常弱或无"
        direction = "正向" if corr_coef > 0 else "负向" if corr_coef < 0 else "无"
        # 解读P值
        if p_value < 0.001: sig_text = "统计上高度显著 (p < 0.001)"
        elif p_value < 0.05: sig_text = "统计上显著 (p < 0.05)"
        else: sig_text = "统计上不显著 (p ≥ 0.05)"
        # 打印解读结论
        print(f"\n[解读] {col1} vs {col2}:")
        print(f"  - 两者之间存在 {strength} 的 {direction} 线性关系。")
        print(f"  - 这种相关性是 {sig_text}。")
        print("-" * 60)
    
    corr = abs(df.select_dtypes(include=['int64', 'float64']).corr())
    lower_triangle = np.tril(corr, k = -1)  
    mask = lower_triangle == 0 

    plt.figure(figsize = (15,8))
    sns.set_style(style = 'white')
    sns.heatmap(lower_triangle, center=0.5, cmap= 'Blues', annot= True, xticklabels = corr.index, yticklabels = corr.columns,
                cbar= False, linewidths= 1, mask = mask)
    plt.xticks(rotation = 50)
    plt.yticks(rotation = 20)
    plt.show()

def is_id_column(column_name):
    """判断一个列名是否是ID列的强硬规则"""
    name = column_name.lower()
    # 规则1: 列名完全等于 'id'
    # 规则2: 列名以下划线加 'id' 结尾 (e.g., 'user_id')
    # 规则3: 列名直接以 'id' 结尾 (e.g., 'userid')
    return name == 'id' or name.endswith('_id') or name.endswith('id')

def skewness_analyze(df: pd.DataFrame, columns2analyze=None):
    """
    对DataFrame中的数值列进行系统性的偏斜程度分析。

    该函数会为每一列数值列生成一个skew图,

    参数:
    ----------
    df : pandas.DataFrame
        包含待分析数据的DataFrame。

    columns2analyze : list of str, optional (默认=None)
        一个包含待分析列名的列表。
        如果为None,函数会自动选择DataFrame中所有的数值类型列。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入 'df' 必须是 pandas DataFrame。")
    
    numerical_df = df.select_dtypes(include=['int64', 'float64'])

    n_cols = 3
    n_rows = (len(numerical_df.columns) // n_cols) + 1

    # Create a figure with subplots
    plt.figure(figsize=(15, 5 * n_rows))  # Adjust size as needed

    # Loop through numerical columns and plot KDE + skewness
    for i, column in enumerate(numerical_df.columns, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.kdeplot(data=numerical_df, x=column, fill=True)
        
        # Calculate skewness
        skewness = skew(numerical_df[column].dropna())  # Handle NaN if needed
        skew_text = f'Skewness: {skewness:.2f}'
        
        # Add skewness as text in the plot
        plt.text(0.05, 0.9, skew_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8))
        
        plt.title(f'KDE of {column}')
        plt.xlabel(column)

    plt.tight_layout()
    plt.show()

def outliers_explorer(df: pd.DataFrame, columns2analyze=None):
    """
    对DataFrame中的数值列进行系统性的异常值探索。

    该函数会为每一列数值列生成一个纺锤图,

    参数:
    ----------
    df : pandas.DataFrame
        包含待分析数据的DataFrame。

    columns2analyze : list of str, optional (默认=None)
        一个包含待分析列名的列表。
        如果为None,函数会自动选择DataFrame中所有的数值类型列。
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("输入 'df' 必须是 pandas DataFrame。")
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    # Plot box plots
    plt.figure(figsize=(15, 8))
    for i, feature in enumerate(numerical_df.columns, 1):
        plt.subplot(2, 4, i)  # Adjust subplot grid as needed
        sns.boxplot(data=df, y=feature, color='skyblue')
        plt.title(f'Box Plot of {feature}')
        plt.tight_layout()
    plt.show()