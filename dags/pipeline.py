from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

import requests
import pickle
import pandas as pd
from bs4 import BeautifulSoup as bs
import datetime as dt
from clearml import Dataset, Task


def custom_stopwords_fr() -> list:
    stopwords_df = pd.read_csv('datasets/stopwords/stopwords_fr.csv')
    custom_stopwords_list = list(stopwords_df['word'])
    return custom_stopwords_list

def scraping_football(nb_page: int):
    goalcom_all_links = []
    for i in range(1, nb_page+1):
        url=f"https://www.goal.com/fr/news/{i}"
        response = requests.get(url)
        html = response.content
        soup = bs(html, "html.parser")
        goalcom_links = ['https://www.goal.com' + a['href'] for a in soup.find_all("a", attrs={'data-testid': 'card-title-url'}) if a.text]
        goalcom_all_links += goalcom_links

    goal_articles = []
    for link in goalcom_all_links:
        try :
            article = []
            response = requests.get(link)
            html = response.content
            soup = bs(html, "html.parser")
            title = soup.select_one('[class*=article_title]').get_text()
            description = soup.select_one('[class*=article_teaser]').get_text()
            article_body = soup.find("div", class_='body')
            article_text = ""
            for a in article_body.find_all(["p", "h2"]):
                if "To read" not in a.text:
                    article_text += " " + a.text
            article = [title, description, article_text]
            goal_articles.append(article)
        except:
            pass

    df = pd.DataFrame(goal_articles, columns=['title', 'description', 'article_text'])
    return df


def scraping_rugby(nb_page: int):
    rugby_all_links = []
    for i in range(1, nb_page):
        url=f"https://www.lerugbynistere.fr/news/page/{i}/"
        response = requests.get(url)
        html = response.content
        soup = bs(html, "html.parser")
        rugby_links = [a['href'] for a in soup.find_all("a", attrs={'class': 'title'}) if a.text]
        rugby_all_links += rugby_links

    rugby_articles = []
    for link in rugby_all_links:
        try :
            article = []
            response = requests.get(link)
            html = response.content
            soup = bs(html, "html.parser")
            
            title = soup.find("h1").get_text()
            title = title.replace('\n', '')
            
            description = soup.find("span", class_='introduction').get_text()

            article_body = soup.find("div", id="article-body")
            for link in article_body.find_all('a', {'class': 'inserer_lien_article'}):
                link.extract()

            article_text = ''
            for paragraph in article_body.find_all('p'):
                article_text += paragraph.text.strip()
            
            article = [title, description, article_text]
            rugby_articles.append(article)
        except:
            pass

    df = pd.DataFrame(rugby_articles, columns=['title', 'description', 'article_text'])
    return df

def scraping_basket(nb_page: int):
    basket_all_links = []
    for i in range(1, nb_page):
        url=f"https://www.parlons-basket.com/category/nba/page/{i}/"
        response = requests.get(url)
        html = response.content
        soup = bs(html, "html.parser")
        basket_links = [a['href'] for a in soup.find_all("a", attrs={'class': 'post-preview-large'}) if a.text]
        basket_all_links += basket_links

    basket_articles = []
    for link in basket_all_links:
        try :
            article = []
            response = requests.get(link)
            html = response.content
            soup = bs(html, "html.parser")
            
            title = soup.find("h1", attrs={'class': 'entry-title'}).get_text()
            
            entry_content_div = soup.find('div', {'class': 'entry-content'})
            description = entry_content_div.find("p").get_text()
            
            p_elements = entry_content_div.find_all('p')
            article_text = ''
            for p in p_elements[1:]:
                if p.get_text() != "Publicité": 
                    article_text += p.text.strip()
            
            article = [title, description, article_text]
            basket_articles.append(article)
        except:
            pass
    
    df = pd.DataFrame(basket_articles, columns=['title', 'description', 'article_text'])
    return df

def scrape_all(nb_articles_par_sujet=200):
    print("football")
    football_df = scraping_football(nb_page=int(nb_articles_par_sujet/30))
    print("rugby")
    rugby_df = scraping_rugby(nb_page=int(nb_articles_par_sujet/15))
    print("basket")
    basket_df = scraping_basket(nb_page=int(nb_articles_par_sujet/10))
    return football_df, rugby_df, basket_df

def creating_full_df(current_date: datetime, basket_df, rugby_df, football_df):
    basket_df['label'] = 'basket'
    rugby_df['label'] = 'rugby'
    football_df['label'] = 'football'

    df = pd.concat([basket_df, rugby_df, football_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(f'datasets/full_articles_{current_date}.csv', index=False)

def creating_model(X_train, y_train):
    from sklearn.naive_bayes import MultinomialNB
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    return nb

def saving_model(model, vectorizer, datetime):
    import os
    if not os.path.exists('models'):
        os.makedirs('models')
    with open(f'models/model_{datetime}.pkl', 'wb') as f:
        pickle.dump((model), f)
    with open(f'models/vectorizer_{datetime}.pkl', 'wb') as f:
        pickle.dump((vectorizer), f)

def load_model(model_path):
    with open(model_path, 'rb') as f:
        nb_model, vectorizer = pickle.load(f)
    return nb_model, vectorizer

def get_model_accuracy(X_test, y_test, model):
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def testing_model(test_phrase: str, vectorizer, model):
    X_new = vectorizer.transform(test_phrase)
    test_phrase_pred = model.predict(X_new)
    print(f'Predictions: {test_phrase_pred}')


def pipeline_scraping_dataset():
    football_df, rugby_df, basket_df = scrape_all(nb_articles_par_sujet=30)
    current_date = dt.datetime.now().strftime('%d-%m')
    creating_full_df(current_date=current_date, football_df=football_df, rugby_df=rugby_df, basket_df=basket_df)

def pipeline_create_clearml_dataset():
    import os
    os.environ['CLEARML_API_HOST'] = "https://api.clear.ml"
    os.environ['CLEARML_WEB_HOST'] = "https://app.clear.ml"
    os.environ['CLEARML_FILES_HOST'] = "https://files.clear.ml"
    os.environ['CLEARML_API_ACCESS_KEY'] = "NXRS736MXOJ50V3NIZD5"
    os.environ['CLEARML_API_SECRET_KEY'] = "ibFYNMzbUiVb8VpRZOZducTBVRUHfpDMWLe4AQE2C4vrsnl4JU"
    current_date = dt.datetime.now().strftime('%d-%m')
    
    task = Task.init(project_name='ArLab Classifier', task_name=f'1 : Create dataset {current_date}')
    
    dataset_name = f'Articles Dataset {current_date}'
    dataset_project = 'Sports Articles Dataset'

    ds = Dataset.create(
        dataset_name=dataset_name,
        dataset_project=dataset_project
    )

    ds.add_files(path='datasets/')
    ds.upload()
    ds.finalize()
    task.mark_completed()


def pipeline_clearml_preprocess_dataset():
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split

    import os
    os.environ['CLEARML_API_HOST'] = "https://api.clear.ml"
    os.environ['CLEARML_WEB_HOST'] = "https://app.clear.ml"
    os.environ['CLEARML_FILES_HOST'] = "https://files.clear.ml"
    os.environ['CLEARML_API_ACCESS_KEY'] = "NXRS736MXOJ50V3NIZD5"
    os.environ['CLEARML_API_SECRET_KEY'] = "ibFYNMzbUiVb8VpRZOZducTBVRUHfpDMWLe4AQE2C4vrsnl4JU"
    
    current_date = dt.datetime.now().strftime('%d-%m')

    task = Task.init(project_name="ArLab Classifier", task_name=f"2 : Preprocess dataset {current_date}")
    df = pd.read_csv(f'datasets/full_articles_{current_date}.csv')

    df = df.dropna()
    X_train, X_test, y_train, y_test = train_test_split(df['article_text'], df['label'], test_size=0.2, random_state=42)
    vectorizer = CountVectorizer(stop_words=custom_stopwords_fr())
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)

    task.upload_artifact('X_train', X_train)
    task.upload_artifact('X_test', X_test)
    task.upload_artifact('y_train', y_train)
    task.upload_artifact('y_test', y_test)
    task.upload_artifact('vectorizer', vectorizer)

    task.mark_completed()

def check_artifacts(ds, **kwargs):
    import os
    os.environ['CLEARML_API_HOST'] = "https://api.clear.ml"
    os.environ['CLEARML_WEB_HOST'] = "https://app.clear.ml"
    os.environ['CLEARML_FILES_HOST'] = "https://files.clear.ml"
    os.environ['CLEARML_API_ACCESS_KEY'] = "NXRS736MXOJ50V3NIZD5"
    os.environ['CLEARML_API_SECRET_KEY'] = "ibFYNMzbUiVb8VpRZOZducTBVRUHfpDMWLe4AQE2C4vrsnl4JU"
    
    current_date = dt.datetime.now().strftime('%d-%m')
    try:
        dataset_task = Task.get_task(project_name="ArLab Classifier", task_name=f"2 : Preprocess dataset {current_date}")
        X_train = dataset_task.artifacts['X_train'].get()
        X_test = dataset_task.artifacts['X_test'].get()
        y_train = dataset_task.artifacts['y_train'].get()
        y_test = dataset_task.artifacts['y_test'].get()
        vectorizer = dataset_task.artifacts['vectorizer'].get()
    except:
        raise ValueError("Some artifacts are missing")

def pipeline_clearml_create_model():
    import joblib
    import os
    os.environ['CLEARML_API_HOST'] = "https://api.clear.ml"
    os.environ['CLEARML_WEB_HOST'] = "https://app.clear.ml"
    os.environ['CLEARML_FILES_HOST'] = "https://files.clear.ml"
    os.environ['CLEARML_API_ACCESS_KEY'] = "NXRS736MXOJ50V3NIZD5"
    os.environ['CLEARML_API_SECRET_KEY'] = "ibFYNMzbUiVb8VpRZOZducTBVRUHfpDMWLe4AQE2C4vrsnl4JU"

    current_date = dt.datetime.now().strftime('%d-%m')
    task = Task.init(project_name="ArLab Classifier", task_name=f"3 : Train model {current_date}")

    dataset_task = Task.get_task(project_name="ArLab Classifier", task_name=f"2 : Preprocess dataset {current_date}")
    X_train = dataset_task.artifacts['X_train'].get()
    X_test = dataset_task.artifacts['X_test'].get()
    y_train = dataset_task.artifacts['y_train'].get()
    y_test = dataset_task.artifacts['y_test'].get()
    vectorizer = dataset_task.artifacts['vectorizer'].get()

    model = creating_model(X_train=X_train, y_train=y_train)
    saving_model(model=model, vectorizer=vectorizer, datetime=current_date)
    accuracy = get_model_accuracy(X_test=X_test, y_test=y_test, model=model)
    task.get_logger().report_scalar("Accuracy", "Score", value=accuracy, iteration=1)
    task.upload_artifact(f'ArLab Model {current_date}',model)
    task.upload_artifact(f'vectorizer {current_date}', vectorizer)

    #joblib.dump(model, f'models/nb_{current_date}.pkl', compress=True)

    task.mark_completed()

with DAG(
    "pipeline_arlab",
    default_args={
        "depends_on_past": True,
        "email": ["airflow@example.com"],
        "email_on_failure": False,
        "email_on_retry": False,
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    description="Pipeline for ArLab",
    schedule=timedelta(minutes=30),
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=["pipeline_arlab"],
) as dag:
    
    scraping_ds = PythonOperator(
        task_id="scraping_ds",
        python_callable=pipeline_scraping_dataset,
    )

    create_ds = PythonOperator(
        task_id="create_ds",
        python_callable=pipeline_create_clearml_dataset,
    )

    preprocess_ds = PythonOperator(
        task_id="preprocess_ds",
        python_callable=pipeline_clearml_preprocess_dataset,
    )

    check_artifacts_task = PythonOperator(
        task_id="check_artifacts",
        provide_context=True,
        python_callable=check_artifacts
    )

    modeling = PythonOperator(
        task_id="modeling",
        python_callable=pipeline_clearml_create_model,
    )

    scraping_ds >> create_ds >> preprocess_ds >> check_artifacts_task >> modeling

