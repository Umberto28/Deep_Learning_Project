# Riconoscimento della Disgrafia con tecniche di Machine Learning e Deep Learning

Questo repository contiene il codice e i materiali relativi ad un progetto sullo sviluppo di modelli di machine learning e deep learning per il riconoscimento automatico della disgrafia a partire da immagini di testo scritto a mano.

## Descrizione del Progetto

La disgrafia è un Disturbo Specifico dell'Apprendimento (DSA) che compromette le abilità di scrittura, rendendo difficoltosa la produzione di testi leggibili e coerenti. Questo progetto mira a supportare la diagnosi della disgrafia attraverso l'uso di tecniche di intelligenza artificiale, automatizzando parte del processo diagnostico.

### Fasi del Progetto

1. **Raccolta dei dati**: Due dataset di immagini di testo scritto a mano sono stati utilizzati per l’addestramento. I dati sono stati etichettati come appartenenti a soggetti con disgrafia o soggetti non disgrafici.

2. **Data Augmentation**: A causa della limitata quantità di dati, sono state applicate tecniche di data augmentation (ritaglio, rotazione, traslazione, capovolgimento orizzontale, aggiunta di rumore gaussiano) per ampliare il dataset e migliorare la generalizzazione dei modelli.

3. **Classificazione con Machine Learning**: È stata eseguita un’estrazione delle caratteristiche utilizzando due reti convoluzionali pre-addestrate (Xception ed EfficientNetB4). Le caratteristiche sono state poi classificate utilizzando diversi algoritmi di machine learning, tra cui Naive Bayes, K-Nearest Neighbors, Random Forest e AdaBoost.

4. **Classificazione con Deep Learning**: È stata utilizzata l'architettura ResNet18, pre-addestrata sul dataset IAM, per la classificazione delle immagini. Il modello è stato successivamente addestrato sui dati di disgrafia, con due scenari di training: uno sui campioni originali e uno con un approccio incrementale sui campioni aumentati.

## Principali Librerie Utilizzate

- `scikit-learn`
- `tensorflow`
- `keras`
- `torch`

## Struttura del Repository

- `/Features`: Contiene le features estratte dalle immagini del dataset nella fase di Machine Learning.
- `/IAM`: Contiene il dataset IAM, utilizzato per il preaddestramento di ResNet18.
- `/dys_dataset`: Contiene i dataset utilizzati per l'addestramento.
- `/src`: Contiene i notebooks utilizzati nella fase di addestramento e gestione dei dati con architetture Deep Learning.
- `/src_ML`: Contiene i notebooks utilizzati nella fase di estrazione delle features e aumento dei dati con architetture Machine Learning.
