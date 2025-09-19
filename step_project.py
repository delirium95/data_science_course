import pandas as pd, requests, os, matplotlib.pyplot as plt, seaborn as sns, numpy as np
from scipy.stats import spearmanr

# --- Завантаження ---
def download_document(file_name, document_url):
    if os.path.exists(file_name):
        print("Файл уже існує, пропускаємо завантаження.")
        return
    response = requests.get(document_url)
    if response.status_code == 200:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print("Файл успішно завантажений!")
    else:
        print(f"Не вдалося завантажити файл. Код статусу: {response.status_code}")

file_name = 'coffee_rates.csv'
document_url = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-07-07/coffee_ratings.csv"
download_document(file_name, document_url)

rates = pd.read_table(file_name, sep=',')

# --- Топ-10 країн експортерів ---
country_export = rates.groupby('country_of_origin')['number_of_bags'].sum().sort_values(ascending=False)
plt.figure(figsize=(12,6))
country_export.head(10).plot(kind='bar', color='brown')
plt.title("Топ-10 країн-експортерів кави (за кількістю мішків)")
plt.ylabel("Сума мішків")
plt.xlabel("Країна")
plt.xticks(rotation=45, ha="right")
plt.show()

# --- Кореляційна матриця Пірсона ---
score_columns = ['aroma', 'flavor', 'aftertaste', 'acidity', 'body', 'balance',
                 'uniformity', 'clean_cup', 'sweetness', 'cupper_points', 'total_cup_points']

corr_matrix = rates[score_columns].corr()
plt.figure(figsize=(12,10))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Кореляції між показниками оцінки кави (Pearson)")
plt.show()

# --- Вплив кольору зерен ---
color_avg = rates.groupby('color')['total_cup_points'].mean().reset_index()
plt.figure(figsize=(6,4))
sns.barplot(x='color', y='total_cup_points', data=color_avg, palette='viridis', ci=None)
plt.title("Вплив кольору зерен на загальну оцінку кави")
plt.ylabel("Середня оцінка")
plt.show()

# --- Середня оцінка по країнах ---
country_score = rates.groupby('country_of_origin')['total_cup_points'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(12,6))
sns.barplot(x='country_of_origin', y='total_cup_points', data=country_score.head(15), palette='Set3')
plt.xticks(rotation=45, ha='right')
plt.title("Середня оцінка кави по країнах")
plt.ylabel("Середня оцінка")
plt.show()

# --- KDE залежність висота-оцінка ---
clean_data = rates.dropna(subset=['altitude_mean_meters', 'total_cup_points'])
clean_data = clean_data[(clean_data['altitude_mean_meters']>=0) & (clean_data['altitude_mean_meters']<=5000)]
plt.figure(figsize=(10,6))
sns.kdeplot(
    x=clean_data['altitude_mean_meters'],
    y=clean_data['total_cup_points'],
    fill=True, cmap="viridis", levels=100
)
plt.axhline(80, color='red', linestyle='--')
plt.axhline(85, color='red', linestyle='--')
plt.axvline(500, color='blue', linestyle='--')
plt.axvline(2000, color='blue', linestyle='--')
plt.xlim(-300, 2500)
plt.ylim(60, clean_data['total_cup_points'].max())
plt.title("Розподіл оцінок кави залежно від висоти")
plt.xlabel("Висота над рівнем моря (метри)")
plt.ylabel("total_cup_points")
plt.show()

# --- Кореляція Спірмена ---
corr, p_value = spearmanr(clean_data['altitude_mean_meters'], clean_data['total_cup_points'])
print(f"Spearman correlation: {corr:.3f}, p-value: {p_value:.3f}")

# --- Фільтрація кореляцій ≥ 0.65 ---
numeric_cols = rates.select_dtypes(include='number').columns
data_numeric = rates[numeric_cols]
corr_matrix = data_numeric.corr(method='pearson')
ignore_cols = ['sweetness', 'uniformity', 'clean_cup']
filtered_cols = []

for col in corr_matrix.columns:
    if col not in ignore_cols:
        if any(corr_matrix[col].abs() >= 0.65):
            filtered_cols.append(col)

corr_filtered = corr_matrix.loc[filtered_cols, filtered_cols]
plt.figure(figsize=(10,8))
sns.heatmap(corr_filtered, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Heatmap of Coffee Attributes (corr ≥ 0.65, без sweetness/uniformity/clean_cup)")
plt.show()

# --- Pivot table для виду і кольору ---
pivot_table = rates.pivot_table(
    index='species',
    columns='color',
    values='total_cup_points',
    aggfunc=np.mean
)

plt.figure(figsize=(8,6))
sns.heatmap(pivot_table, annot=True, fmt=".1f", cmap="YlGnBu")
plt.title("Середні оцінки кави по виду та кольору зерен")
plt.show()

# --- Барплот за видом і кольором ---
plt.figure(figsize=(10,6))
sns.barplot(x='species', y='total_cup_points', hue='color', data=rates, palette='Set2', ci=None)
plt.title("Середні оцінки кави за видом і кольором зерен")
plt.ylabel("Середня оцінка")
plt.show()

# --- Scatter plot вид-колір ---
scatter_data = pivot_table.reset_index().melt(
    id_vars='species',
    value_vars=pivot_table.columns,
    var_name='color',
    value_name='avg_score'
)
plt.figure(figsize=(8,6))
sns.scatterplot(
    x='color',
    y='avg_score',
    hue='species',
    data=scatter_data,
    s=100,
    palette='Set1'
)
plt.title("Вплив кольору зерен на рейтинг кави")
plt.ylabel("Середня оцінка")
plt.show()
