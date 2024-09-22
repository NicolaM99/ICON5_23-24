# Progetto Esame ICON5_23-24
---

# Heart Disease Analysis - ICON5_23-24

**Componenti del gruppo**
- Nicola Mastromarino Matricola: 757709, n.mastromarino5@studenti.uniba.it

**Link GitHub:** [https://github.com/NicolaM99/ICON5_23-24](https://github.com/NicolaM99/ICON5_23-24)

**Anno Accademico:** 2023-2024

---

## Indice

1. **[Introduzione](#1-introduzione)**
   - [1.1 Scopo del Progetto](#11-scopo-del-progetto)
   - [1.2 Obiettivi](#12-obiettivi)
   - [1.3 Struttura del Documento](#13-struttura-del-documento)
   
2. **[Specifiche Tecniche](#2-specifiche-tecniche)**
   - [2.1 Requisiti Funzionali](#21-requisiti-funzionali)
   - [2.2 Ambiente di Sviluppo](#22-ambiente-di-sviluppo)
   - [2.3 Librerie e Strumenti Utilizzati](#23-librerie-e-strumenti-utilizzati)

3. **[Fondamenti Teorici](#3-fondamenti-teorici)**

4. **[Preparazione dei Dati](#4-preparazione-dei-dati)**

5. **[Ingegneria delle Caratteristiche](#5-ingegneria-delle-caratteristiche)**

6. **[Modelli di Apprendimento Automatico](#6-modelli-di-apprendimento-automatico)**

7. **[Gestione dello Sbilanciamento dei Dati](#7-gestione-dello-sbilanciamento-dei-dati)**

8. **[Clustering e Analisi Esplorativa](#8-clustering-e-analisi-esplorativa)**

9. **[Modelli di Insieme](#9-modelli-di-insieme)**

10. **[Rete Bayesiana](#10-rete-bayesiana)**

11. **[Risultati e Discussione](#11-risultati-e-discussione)**

12. **[Riferimenti](#12-riferimenti)**

---

### Abstract

Questo progetto analizza e prevede il rischio di malattie cardiache utilizzando tecniche avanzate di apprendimento automatico e modellazione probabilistica. Il sistema affronta diverse sfide riguardanti il preprocessing dei dati, l'ingegneria delle caratteristiche e la gestione dello sbilanciamento delle classi. Include tecniche di classificazione, clustering e inferenze probabilistiche con reti bayesiane.

---

## 1. Introduzione

### 1.1. Scopo del Progetto

L'obiettivo è migliorare la capacità predittiva riguardante le malattie cardiache utilizzando tecniche di apprendimento automatico e modellazione probabilistica su un dataset specifico. Il progetto mira a sviluppare un sistema predittivo per supportare decisioni cliniche.

### 1.2. Obiettivi

- Caricamento e preprocessing dei dati.
- Creazione di caratteristiche polinomiali e selezione delle caratteristiche più rilevanti.
- Implementazione di vari modelli di classificazione ottimizzati.
- Gestione dello sbilanciamento dei dati con SMOTE.
- Applicazione di clustering e reti bayesiane per inferenze probabilistiche.

### 1.3. Struttura del Documento

Il documento è suddiviso in sezioni che coprono le fasi di preprocessing, ingegneria delle caratteristiche, modellazione, clustering, e reti bayesiane.

---

## 2. Specifiche Tecniche

### 2.1. Requisiti Funzionali

Il sistema deve:
- Eseguire preprocessing dei dati.
- Applicare tecniche di ingegneria delle caratteristiche.
- Implementare modelli di classificazione e ottimizzazione degli iperparametri.
- Gestire sbilanciamenti delle classi.

### 2.2. Ambiente di Sviluppo

- **Linguaggio:** Python 3.x
- **IDE:** PyCharm
- **Librerie:** Pandas, Scikit-learn, XGBoost, Imbalanced-learn, Matplotlib, Seaborn, pgmpy

### 2.3. Librerie e Strumenti Utilizzati

- **Pandas** per gestione dei dati.
- **Scikit-learn** per preprocessing e modelli di machine learning.
- **XGBoost** per boosting.
- **Imbalanced-learn** per bilanciare i dati.
- **pgmpy** per reti bayesiane.

---

## 3. Fondamenti Teorici

Il progetto applica tecniche di apprendimento supervisionato, non supervisionato, e reti bayesiane per modellare le relazioni tra variabili.

---

## 4. Preparazione dei Dati

Viene effettuata una pre-elaborazione completa del dataset, includendo gestione dei valori mancanti, normalizzazione, e trasformazione delle variabili target.

---

## 5. Ingegneria delle Caratteristiche

Il sistema genera caratteristiche polinomiali e utilizza tecniche come l'Eliminazione Ricorsiva delle Caratteristiche (RFE) per migliorare le prestazioni dei modelli.

---

## 6. Modelli di Apprendimento Automatico

Sono stati implementati diversi modelli di classificazione, tra cui Random Forest, k-NN, XGBoost, Decision Tree, e Naive Bayes, ottimizzati con GridSearchCV.

---

## 7. Gestione dello Sbilanciamento dei Dati

Il sistema utilizza SMOTE per bilanciare le classi del dataset e migliorare la capacità del modello di predire correttamente la classe minoritaria.

---

## 8. Clustering e Analisi Esplorativa

Sono stati utilizzati algoritmi come K-means e DBSCAN per identificare gruppi di pazienti con caratteristiche simili.

---

## 9. Modelli di Insieme

Un Voting Classifier combina diversi modelli per migliorare la robustezza e le prestazioni complessive del sistema.

---

## 10. Rete Bayesiana

La rete bayesiana costruita permette di modellare le relazioni probabilistiche tra variabili cliniche e di effettuare inferenze.

---

## 11. Risultati e Discussione

### 11.1. Sommario dei Risultati

I modelli sviluppati hanno ottenuto risultati robusti, con l'uso di SMOTE e delle tecniche di ingegneria delle caratteristiche che ha migliorato le predizioni.

### 11.2. Considerazioni Finali

Il progetto dimostra l'efficacia delle tecniche di machine learning e reti bayesiane nell'analisi predittiva delle malattie cardiache.

### 11.3. Lavori Futuri

Lavori futuri potrebbero includere l'implementazione di deep learning e la validazione clinica dei modelli sviluppati.

---

## 12. Riferimenti

- UCI Machine Learning Repository: [Cleveland Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
  
