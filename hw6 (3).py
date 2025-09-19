import pandas as pd, requests, os


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


def read_file(file_name):
    if not os.path.exists(file_name):
        print("Файл не знайдено.")
        return []
    with open(file_name, 'r', encoding='utf-8') as f:
        content = f.read()
    return content


def save_info(file_name, info):
    with open(file_name, 'r') as f:
        f.write(info)
        print("Successfully saved.")


file_name = 'movies.csv'
document_url = 'https://gist.githubusercontent.com/tiangechen/b68782efa49a16edaf07dc2cdaa855ea/raw/0c794a9717f18b094eabab2cd6a6b9a226903577/movies.csv'
download_document(file_name, document_url)

movies = pd.read_csv('movies.csv')
print(movies.dtypes) #2
print(movies.info())
print(movies.describe())
print(movies["Profitability"].describe())
(movies["Worldwide Gross"].describe())
print(movies["Year"].describe())
print(movies["Rotten Tomatoes %"].describe())
print(movies["Genre"].value_counts())
print(movies.shape[0]) # movie count
print(movies["Year"].value_counts().sort_index())
print(movies.groupby("Year")["Film"].count())
print(movies.nlargest(5, 'Profitability'))
print(movies.nsmallest(5, 'Profitability'))
movies['Genre'] = movies['Genre'].replace({
    'romance': 'Romance',
    'comedy': 'Comedy',
    'Romence': 'Romance',
    'Comdy': 'Comedy',
})
comedies = movies[movies['Genre'].str.lower() == 'comedy']
top_10_comedies = comedies.sort_values(by='Audience score %', ascending=False).head(10)
top_10_comedies_selected = top_10_comedies[['Film', 'Year', 'Lead Studio']]
top_10_comedies_selected.to_csv('top_10_comedies_selected.csv', index=False)