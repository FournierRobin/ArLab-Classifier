# ArLab Classifier
Ce projet vise à entraîner un modèle de classification de texte à l'aide d'Airflow et ClearML, et à le déployer via une API FastAPI et une interface utilisateur Streamlit.

### Liens utiles
- **Airflow** : http://localhost:8080/dags
- **ClearML** : https://app.clear.ml
- **API** : http://localhost:8000/predict
- **Streamlit** : http://localhost:8501

### Description du projet
On scrape d'abord 3 sites respectivement de :
- Football
- Rugby
- Basket

Puis on preprocess le dataset et entraine un modèle de classification de texte.
Pour cela, on utilise des tâches ClearML afin des suivres les métriques d'entrainement et on lance une pipeline de ces tâches avec Airflow pour automatiser le tout.
Le dataset et le modèle entraîné sont ensuite enregistrés en tant que Dataset et Artefact ClearML.

On utilise une API FastAPI pour pouvoir requeter notre model et obtenir nos prédictions, le tout habillé dans un UI Streamlit.

### Configuration du projet
Pour lancer l'application, il faut exécutez les commandes suivantes :

`git clone https://github.com/IA-cloud-Ynov-2023/ArLab`

`docker-compose up --build`