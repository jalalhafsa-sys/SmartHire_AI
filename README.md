SmartHire-AI — Classification des CV et matching sémantique CV–offres
Projet académique — Master Sciences des Données et Intelligence Artificielle (SDIA)

SmartHire-AI est une application de recrutement intelligent développée avec Python et Streamlit. Elle propose deux parcours complémentaires pour faciliter l'exploration de CV et d'offres d'emploi. Les recommandations constituent une aide à la présélection, et non une décision automatique de recrutement.
1. Objectifs
- Espace Recruteur : saisir une offre d'emploi libre (intitulé, description, compétences) et obtenir les 5 CV les plus similaires parmi 2 481 profils.
- Espace Candidat : importer un CV (PDF ou image) ou coller son texte, obtenir une classification professionnelle par SVM et découvrir les 5 offres les plus similaires parmi 14 073 offres.
- Proposer une interface interactive permettant de consulter les résultats, explorer le catalogue et visualiser des analyses.
Important : les scores affichés sont des similarités cosinus, et non des probabilités d'embauche ni des validations automatiques des compétences.

2. Technologies utilisées
Composant	Technologie
Langage et application	Python, Streamlit
Classification des CV	Scikit-learn, TF-IDF, SVM linéaire (LinearSVC)
Matching sémantique	sentence-transformers/all-MiniLM-L6-v2
Similarité	Similarité cosinus sur embeddings normalisés
Extraction de documents	PyMuPDF, OCR Tesseract local
Visualisation	Plotly
Expérimentation	Notebook Kaggle


3. Données et préparation
Le projet exploite 2 481 CV, organisés en 24 catégories professionnelles, ainsi qu'un catalogue de 14 073 offres d'emploi. Les corpus sont principalement anglophones.

La préparation des textes inclut notamment le nettoyage HTML, la normalisation Unicode NFKC, la gestion des espaces et un masquage partiel de certaines coordonnées. Pour MiniLM, les textes longs sont découpés en segments de 200 tokens, encodés par lots de 16, puis agrégés par moyenne et normalisés en L2.

L'anonymisation est partielle : elle ne garantit pas la suppression de toutes les données personnelles.
4. Modèles et méthodologie
Classification des CV — SVM
La classification utilise une représentation TF-IDF et un SVM linéaire (LinearSVC) afin de prédire l'une des 24 catégories professionnelles. Le modèle optimisé utilise notamment C=10, class_weight="balanced", max_iter=5000 et random_state=42.

La séparation entraînement/test est de 80 % / 20 % (soit 1 984 CV / 497 CV). L'optimisation s'appuie sur une validation croisée stratifiée à 5 plis et le F1-score macro.
Matching sémantique — MiniLM
Le modèle préentraîné all-MiniLM-L6-v2 transforme les textes en vecteurs de 384 dimensions. Le classement repose sur la similarité cosinus, calculée entre une nouvelle requête et les embeddings préencodés du corpus correspondant.
- Recruteur : nouvelle offre → comparaison avec les 2 481 embeddings CV → Top 5 CV.
- Candidat : CV analysé → comparaison avec les 14 073 embeddings d'offres → Top 5 offres.
La recherche porte sur l'ensemble du corpus : la catégorie prédite par le SVM ne constitue pas un filtre implicite du matching.
5. Résultats de classification
Métrique	SVM initial	SVM optimisé
Accuracy	70,02 %	71,03 %
F1-score macro	66,43 %	67,57 %
F1-score pondéré	68,77 %	69,67 %


Ces valeurs concernent l'évaluation de la classification SVM, et non la pertinence du matching MiniLM. Le matching nécessite une évaluation métier distincte sur des paires CV–offre annotées.
6. Fonctionnalités de l'application
Espace Recruteur
1. Ouvrir Espace Recruteur.
2. Renseigner l'intitulé, la description et les compétences du poste (de préférence en anglais, comme le corpus).
3. Cliquer sur Rechercher les CV pertinents.
4. Consulter les 5 profils classés par similarité décroissante.
Les CV du corpus sont présentés au moyen d'alias et d'aperçus limités. La catégorie affichée provient des données du corpus, et non d'une nouvelle prédiction SVM.
Espace Candidat
1. Ouvrir Espace Candidat.
2. Importer un PDF, PNG, JPG ou JPEG, ou coller le texte d'un CV.
3. Lancer Analyser mon CV.
4. Consulter la classification et les 5 offres recommandées.
5. Explorer les sections de classification, matching, catalogue et analyses.
L'extraction prend en charge les PDF textuels et l'OCR local en français et anglais, lorsque les dictionnaires requis sont présents.
7. Architecture du projet
SmartHire-AI/
├── app/
│   ├── app.py             # Interface Streamlit et orchestration
│   ├── nlp.py             # Moteur NLP
│   ├── recruiter.py       # Recherche de CV côté recruteur
│   ├── recruiter_ui.py    # Interface recruteur
│   ├── extraction.py      # Extraction PDF/image et OCR
│   ├── navigation.py      # Navigation
│   ├── components.py      # Composants d'interface
│   ├── catalogue.py       # Consultation des offres
│   ├── charts.py          # Graphiques
│   └── styles.css         # Styles
├── models/                # Modèles, données et ressources locales
├── notebooks/             # Notebook Kaggle
├── tests/                 # Tests techniques
├── requirements.txt
└── README.md
8. Installation et exécution
Prérequis
- Python compatible avec les dépendances du projet (environnement documenté : Python 3.10 à 3.13).
- Les fichiers modèles et données exportés depuis Kaggle.
- Une connexion Internet au premier chargement de MiniLM, sauf si ses poids sont déjà présents dans le cache local.
Sur un nouvel ordinateur (PowerShell)
Depuis le dossier du projet :
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
Ouvrir l'adresse locale affichée par Streamlit (généralement http://localhost:8501).

Remarque : les artefacts SVM du projet exigent scikit-learn 1.6.1. Les modèles et datasets ne sont pas nécessairement présents dans le dépôt GitHub : ils doivent être fournis séparément et placés aux emplacements attendus. Les fichiers joblib doivent provenir d'une source de confiance.
Sur le poste de développement déjà configuré
cd C:\SmartHire-AI
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -B -m streamlit run app/app.py --server.address=127.0.0.1 --server.port=8501
Ce second mode suppose que .venv, les données et les poids MiniLM sont déjà installés sur la machine locale.
9. Vérification technique
Le projet dispose de tests couvrant notamment l'extraction des documents, les parcours Streamlit, le classement des résultats, les filtres et l'inférence des modèles. La documentation technique initiale rapporte 51 tests réussis, aucun ignoré, lors de la validation du 26 octobre 2026.

Une vérification d'alignement numérique des embeddings exportés est également documentée pour 2 481 CV et 14 073 offres. Ces contrôles vérifient le fonctionnement et la cohérence des artefacts : ils ne démontrent pas la pertinence métier des recommandations.

Pour exécuter les tests sur l'environnement de développement configuré :
$env:SMARTHIRE_LIVE_MODEL_TEST='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:KMP_DUPLICATE_LIB_OK='FALSE'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp=.cache/recruiter-tests
10. Limites et perspectives
- Classification imparfaite : certaines catégories professionnelles peuvent être confondues.
- Pertinence métier à valider : un score de similarité élevé ne garantit ni l'adéquation complète au poste ni la présence de toutes les compétences exigées.
- Langue des données : le corpus étant principalement anglophone, la qualité sur des documents français doit être évaluée séparément.
- Confidentialité et équité : la protection des données personnelles et l'évaluation des biais nécessitent une attention particulière.
- Décision humaine : les résultats sont des recommandations d'aide à la sélection, jamais une décision d'embauche automatisée.
11. Dépôt et ressources
Le dépôt GitHub contient le code et/ou les archives effectivement publiés par l'autrice du projet. Les gros artefacts, caches, environnements et données sensibles peuvent être exclus du suivi Git ; leur disponibilité doit être vérifiée avant toute reproduction.

Projet réalisé dans le cadre du Master SDIA.
