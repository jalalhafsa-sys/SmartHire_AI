# SmartHire-AI — Matching sémantique CV–Offres d’emploi

Projet de Master SDIA avec deux parcours complémentaires :

- **Recruteur (principal)** : offre libre → MiniLM → Top 5 CV parmi 2 481 profils.
- **Candidat** : CV PDF/image/texte → classification SVM → Top 5 parmi 14 073 offres.

L’interface Indigo & Lavender propose un accueil, les deux espaces et les pages
complémentaires de classification, matching, catalogue, analyses et méthodologie.
Les scores sont des similarités cosinus, jamais des probabilités d’embauche.

## Exécuter l’application existante

L’environnement `.venv` et les poids MiniLM sont déjà installés sur ce poste.
Dans PowerShell :

```powershell
cd C:\SmartHire-AI
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
.\.venv\Scripts\python.exe -B -m streamlit run app/app.py --server.address=127.0.0.1 --server.port=8501
```

Ouvrir **http://127.0.0.1:8501**. Aucun entraînement n’est déclenché.
Ne recréez pas l’environnement pour simplement relancer l’application.

Sur un autre poste, utiliser Python 3.10–3.13, créer un environnement virtuel puis
installer `requirements.txt`. Les artefacts exigent scikit-learn **1.6.1**. Fournir
les six exports Kaggle décrits ci-dessous ; seuls des artefacts de confiance doivent
être désérialisés avec joblib. Le premier chargement de MiniLM nécessite une
connexion si ses poids ne sont pas déjà en cache dans `models/hf_cache/`.

## Démonstration — Espace Recruteur

1. Ouvrir « Espace Recruteur » ou cliquer sur « Rechercher des CV » dans l’accueil.
2. Saisir l’intitulé, la description et les compétences recherchées. Le poste n’a
   pas besoin d’exister dans le catalogue LinkedIn.
3. Cliquer sur « Rechercher les CV pertinents ». Seule cette nouvelle offre est
   encodée : les 2 481 embeddings CV existants sont chargés et réutilisés.
4. Consulter les cinq profils, leur catégorie du corpus et leurs cosinus décroissants.
5. Modifier les champs puis relancer pour remplacer les résultats. Une recherche
   invalide retire les anciens résultats ; le parcours candidat conserve son état.

Les noms, coordonnées, identifiants originaux et textes intégraux du corpus CV ne
sont pas affichés. Les cartes utilisent des alias de position `Profil CV-0001` et
une **liste fermée de termes professionnels** repérés dans les textes. C’est un
aperçu limité, pas une extraction exhaustive ni une validation des compétences.
La catégorie présentée vient du CSV et n’est pas une nouvelle prédiction SVM.

Les fichiers ajoutés doivent rester ensemble et dans leur ordre d’export Kaggle :
`models/cvs.csv` (colonnes réelles : `cv_id`, `category`, `texte_final`) et
`models/embeddings_cvs.npy` (float32, 2481 × 384, L2). Le chargeur vérifie dimensions,
valeurs finies, unicité des identifiants, catégories et normalisation. Il ne trie
et ne déduplique aucune ligne. Le corpus CV privé reste en cache de ressources
mémoire, sans cache disque ni transmission au navigateur. Le catalogue d’offres
conserve son cache de données ; l’encodeur MiniLM partagé reste protégé par verrou.

La requête assemble les champs non vides avec des espaces et applique les fonctions
Kaggle existantes (anonymisation partielle, nettoyage HTML, NFKC). Les segments de
200 tokens sont encodés comme dans le Notebook, puis moyennés et normalisés L2.
Les cosinus portent sur tous les CV. Le tri est stable ; les égalités conservent
l’ordre d’export. Aucune catégorie professionnelle ne restreint les résultats.

Architecture ajoutée : `app/recruiter.py` (chargement, contrôles, recherche, projection
sécurisée), `app/recruiter_ui.py` (formulaire et cartes), `tests/test_recruiter.py`.
Le moteur candidat `app/nlp.py` et les six fichiers modèles/données restent inchangés.
Ce corpus CV n’est pas, à lui seul, un jeu de test indépendant : aucune métrique de
performance n’est déduite des résultats de recherche.

## Démonstration — Espace Candidat

1. Depuis l’accueil, ouvrir « Espace Candidat · Analyser mon CV ».
2. Dans l’onglet par défaut, importer un PDF, PNG, JPG ou JPEG. L’extraction est
   automatique ; ouvrir l’aperçu du texte pour vérifier le contenu.
3. Cliquer sur « Analyser mon CV » pour obtenir la catégorie réelle et le Top 5.
4. Explorer Classification NLP, Matching et Analytics : les résultats restent
   disponibles pendant la navigation.
5. Filtrer le matching par intitulé, expression littérale et cosinus minimal.
   Le catalogue possède une recherche indépendante et une pagination de 10 offres.
6. « Réinitialiser l’analyse » efface résultats, texte extrait et entrées de session.

Le second onglet analyse uniquement le texte collé. Il n’utilise jamais le fichier
importé ; réciproquement, l’analyse du fichier ignore le texte collé. Les cartes
présentent les descriptions et les termes TF-IDF partagés. Ces indices lexicaux
ne constituent pas une explication causale du score MiniLM.

## Extraction et OCR local

- PDF textuels : PyMuPDF, toutes les pages.
- PDF scannés : reconnaissance des pages sans texte exploitable ; les PDF mixtes
  peuvent aussi faire reconnaître leurs images en conservant la couche texte.
- Images : conversion PDF en mémoire et OCR Tesseract intégré à PyMuPDF.
- Langues : `fra+eng`. Les dictionnaires officiels **tessdata_fast** sont dans
  `models/tessdata/fra.traineddata` et `eng.traineddata` (environ 5 Mo au total).
  Aucun service OCR externe ni installation supplémentaire de PyTorch.
- Limites : fichier 10 Mo, PDF 50 pages, texte 150 000 caractères et au moins
  20 lettres exploitables. Les dimensions extrêmes et fichiers illisibles sont rejetés.
- Si les dictionnaires manquent, un scan ou une image affiche explicitement
  « OCR local indisponible ». Un PDF textuel reste utilisable.

Documentation : [OCR PyMuPDF](https://pymupdf.readthedocs.io/en/latest/recipes-ocr.html),
[dictionnaires officiels Tesseract](https://github.com/tesseract-ocr/tessdata_fast).
Les erreurs OCR et la qualité sur les mises en page complexes nécessitent une
vérification humaine du texte extrait avant l’analyse.

## Architecture

```text
app/
  app.py           Interface, orchestration et caches Streamlit
  nlp.py           Moteur Kaggle existant, inchangé
  extraction.py    Extraction PDF/image et OCR en mémoire
  navigation.py    Sept sections et réinitialisation
  components.py    Cartes, états vides et éléments communs
  catalogue.py     Filtres et pagination sans mutation
  charts.py        Graphiques Plotly interactifs
  styles.css       Identité Indigo & Lavender responsive
models/            Exports privés, caches MiniLM et dictionnaires OCR
notebooks/         Notebook Kaggle original, inchangé
tests/            Parité Kaggle, extraction, UI et inférence réelle
```

`st.cache_resource` conserve les modèles et un verrou protège l’encodeur partagé.
`st.cache_data` conserve uniquement le catalogue public au projet et ses agrégats.
Les données personnelles restent dans `st.session_state`, jamais dans ces caches.

## Fidélité au Notebook Kaggle

Les fichiers `models/tfidf.joblib`, `svm_optimise.joblib`, `jobs.csv` et
`embeddings_jobs.npy` restent inchangés et doivent provenir du même export.
Le catalogue possède exactement **14 073 offres** et les colonnes `job_id`,
`title`, `description`, `texte_final`. Le classifieur possède **24 catégories**.
Aucun salaire, lieu ou catégorie d’offre n’est inventé.

Fonctions Kaggle originales (cellules 16–18 et 28–29) : masquage email, LinkedIn,
téléphone ; nettoyage HTML ; Unicode NFKC et espaces ; segmentation sans
chevauchement en **200 tokens** ; encodage par lots de 16 ; moyenne des embeddings
de segments et normalisation **L2** du document. Le module `Normalize` natif de
`sentence-transformers/all-MiniLM-L6-v2` reste actif comme dans Kaggle.

Le cosinus compare le document aux **14 073 × 384** embeddings normalisés.
L’ordre du catalogue et des vecteurs est conservé ; les filtres ne modifient que
la vue. Les égalités gardent l’ordre source. La catégorie SVM n’est pas un filtre
implicite. Les scores sont des cosinus sur **[−1, 1]**, jamais des probabilités.
Une barre représente la partie positive ; un cosinus négatif reste affiché signé.

Les contrôles de dimensions ne prouvent pas à eux seuls un alignement historique
complet. Les tests réencodent les offres première, médiane et dernière et les
comparent aux exports. Aucun modèle n’est réentraîné ni artefact réécrit.

## Tests

```powershell
cd C:\SmartHire-AI
$env:SMARTHIRE_LIVE_MODEL_TEST='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:KMP_DUPLICATE_LIB_OK='FALSE'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --tb=short
.\.venv\Scripts\python.exe -B -m pip check
```

Sans `SMARTHIRE_LIVE_MODEL_TEST=1`, les tests explicitement marqués MiniLM réel
sont ignorés. Les autres tests continuent de vérifier les vrais artefacts SVM,
la parité des fonctions Kaggle et l’extraction. Les documents de test sont créés
en mémoire ; les parcours NLP utilisent le texte réel d’une offre comme document
technique, sans inventer de CV personnel ni de scores utilisateur.

AppTest vérifie les sept pages, graphiques Plotly, filtres, pagination, état de
session et erreurs. Pour les fichiers, seul le widget d’import est remplacé par
un objet de test : extraction, OCR et inférence sont réellement exécutés. Le geste
de glisser-déposer dans un navigateur reste une vérification manuelle distincte.
Les tests ne constituent pas une évaluation de pertinence métier.

## Environnement Windows local

La `.venv` existante réutilise les bibliothèques Anaconda via
`--system-site-packages`. Son fichier local `00_smarthire_runtime.pth` sélectionne
`MKL_THREADING_LAYER=SEQUENTIAL`, `USE_TF=0` et `USE_TORCH=1` pour éviter le conflit
OpenMP et l’import de TensorFlow/Keras 3. Aucune configuration globale n’est modifiée.
Les caches HF, Torch, Matplotlib et IPython sont dirigés dans ce projet.

## Confidentialité et limites

Les CV ne sont ni écrits sur disque ni journalisés ni transmis à un service OCR.
Ils restent en mémoire de session Streamlit jusqu’à réinitialisation ou fermeture
et libération de la session. L’anonymisation Kaggle n’efface pas toutes les données
personnelles, notamment les noms. Le corpus est principalement anglophone :
la qualité sur des CV français et tout seuil de pertinence restent à valider.
Aucune décision d’embauche ne doit être déduite automatiquement des recommandations.

## Git et GitHub

Les modèles, dictionnaires OCR, données, résultats, caches et environnements sont
ignorés par `.gitignore`. Aucune publication n’est effectuée. Inspecter aussi les
sorties du Notebook avant toute publication autorisée. Étapes restantes après
accord : vérifier `git status`, créer un commit des seules sources approuvées,
créer le dépôt distant puis ajouter `origin` et pousser la branche choisie.

## Validation de la refonte ? 25 octobre 2026

Les 35 tests ont r?ussi avec les poids r?els MiniLM et sans test ignor?. OCR PNG,
JPG, JPEG, PDF scann?, PDF textuel multipage, sept sections, filtres, pagination,
conservation des r?sultats, graphiques et r?initialisation ont ?t? v?rifi?s.
`pip check` ne signale aucune d?pendance incoh?rente. Les captures Chrome et le
compte rendu sont dans `results/`, exclus de Git. Aucun entra?nement ni publication.

Un avertissement non bloquant du tokenizer peut mentionner un texte brut sup?rieur
? 256 tokens : la fonction Kaggle le d?coupe ensuite en blocs de 200 tokens avant
l?encodage. Ce calcul original est conserv? ; les tests de parit? et les inf?rences
r?elles r?ussissent. La pertinence sur un CV personnel et la qualit? OCR de toute
mise en page ne sont pas garanties par ces tests techniques.

## Validation des deux parcours — 26 octobre 2026

La suite complète a réussi : **51 tests, aucun ignoré**, avec le SVM et MiniLM réels.
Les tests couvrent trois métiers du catalogue (développement logiciel, comptabilité,
enseignement), une offre libre hors catalogue, le tri exact sur les 2 481 embeddings
CV, le remplacement des recherches, les erreurs, les aperçus contrôlés et le maintien
du parcours candidat. Les tests OCR et d’évaluation précédents passent également.
Chrome a permis de vérifier le formulaire et le Top 5 réel, la navigation candidat,
et le rendu desktop/mobile. Captures : `results/recruiter-home.png`,
`recruiter-results.png`, `recruiter-mobile.png`, exclus de Git.

Les résultats de ces tests sont fonctionnels : ils ne constituent ni une mesure de
pertinence métier, ni un taux de réussite de recrutement. Aucune métrique inventée.
Les six fichiers modèles/données, le Notebook et `app/nlp.py` sont conservés.

Pour reproduire tous les tests :

```powershell
cd C:\SmartHire-AI
$env:SMARTHIRE_LIVE_MODEL_TEST='1'
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:KMP_DUPLICATE_LIB_OK='FALSE'
.\.venv\Scripts\python.exe -B -m pytest -q -p no:cacheprovider --basetemp=.cache/recruiter-tests
```

## Alignement exhaustif — 26 octobre 2026

Chaque ligne a été réencodée une fois hors interface, puis comparée au vecteur
exporté au même indice : **2 481 / 2 481 CV et 14 073 / 14 073 offres conformes**.
Aucun vecteur n’a été écrit ni remplacé. Les empreintes SHA-256 des six fichiers
modèles/données sont inchangées. Le contrôle utilise MiniLM, les segments de
200 tokens, `batch_size=16`, la moyenne par document et la normalisation L2.
La tolérance numérique de comparaison est `atol=1e-4, rtol=1e-3`.
Le détail est dans `results/recruiter-full-alignment.json` et la validation globale
dans `results/recruiter-validation.json`. Ce contrôle numérique d’alignement n’est
pas une mesure de pertinence des recommandations. Il n’est jamais exécuté lors
d’une recherche dans l’application : seuls les nouveaux textes saisis sont encodés.
