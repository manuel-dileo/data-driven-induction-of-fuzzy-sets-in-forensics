# Data-driven-induction-of-fuzzy-sets-in-forensics
Bachelor's Thesis in Computer Science,  
Advisor: Prof. Dario Malchiodi,  
University of Milan, 2020.
## Informazioni generali
Questo repository ha lo scopo di illustrare il codice realizzato durante l'esperienza di tirocinio interno triennale presso il dipartimento d'informatica dell'Università Degli Studi di Milano. Il lavoro svolto è descritto e commentato all'interno della tesi di laurea "Induzione di insiemi fuzzy in ambito medico-legale".  
Il lavoro descrive l’utilizzo dell’induzione di insieme fuzzy per la classificazione di dati in un problema di medicina legale. Il problema consiste nel classificare correttamente delle persone, ovvero delle osservazioni di un campione di cui conosciamo caratteristiche anagrafiche e riguardanti le lesioni sul corpo subite in un incidente, come investite da mezzi pesanti o mezzi leggeri.
## Specifiche Hardware
Gli esperimenti sono stati eseguiti con un computer portatile hp modello 15-aw010nl che possiede le seguenti caratteristiche:
- Processore AMD Quad-Core A10-9600P (2,4 GHz, fino a 3,3 GHz, 2 MB di cache).
- Memoria 16 GB di SDRAM DDR4-2133 (2 x 8 GB).
- Scheda Grafica AMD Radeon™ R7 M440 (DDR3 da 2 GB dedicata).  

La computazione dell'addestramento del modello di apprendimento, data una singola configurazione di iperparametri, impiega mediamente 1 minuto e mezzo.
## Descrizione dei notebook
All'interno di questo repository sono caricati diversi jupyter notebook. Ognuno di essi fa riferimento a una specifica parte del lavoro e/o una specifica vista del dataset su cui sono stati svolti degli esperimenti. In particolar modo:
- experiments\_all-feature.ipynb contiene gli esperimenti svolti sul dataset originario.
- first\_experiments\_20feature.ipynb contiene i primi esperimenti svolti durante l'esperienza del tirocinio. Il notebook è stato inserito con il solo scopo di illustrare i primi passi del mio lavoro e presenta pertanto esperimenti eseguiti con griglie più semplici e in ordine sparso. Sconsiglio l'esecuzione di questo notebook sui propri dispositivi.
- experiments\_20feature.ipynb contiene gli esperimenti più significativi sulla vista del dataset ottenuta considerando 20 variabili scelte dai medici legali.
- experiments\_13feature.ipynb contiene gli esperimenti più significativi sulla vista del dataset ottenuta considerando 13 variabili scelte dai medici legali.
- experiments\_datasetWithFeatureExtraction.ipynb contiene gli esperimenti svolti su una vista del dataset costruita aggregando le variabili delle lesioni per zone sensate, utilizzando una tecnica di riduzione della dimensionalità come PCA.
- experiments\_dxsx.ipynb contiene gli esperimenti con le viste del dataset costruite a partire da una suddivisione delle lesioni sulla parte destra o sinistra del corpo.
- experiments\_umap.ipynb contiene gli esperimenti in cui viene utilizzata UMAP come tecnica di riduzione della dimensionalità.
- data\_augmentation.ipynb illustra due tecniche per sovracampionare il dataset al fine di migliorare l'accuratezza del modello.
- datavis.ipynb illustra una possibile visualizzazione del dataset.
- exploratory\_analysis\_for\_defuzzification.ipynb presenta l'analisi esplorativa svolta al fine di defuzzificare in modo sensato i risultati.
- defuzzification\_20feature.ipynb contiene la defuzzificazione dei risultati del miglior esperimento sulla vista ottenuta considerando 20 variabili scelte dai medici legali.  

Nel caso foste interessati esclusivamente alla lettura dei migliori risultati del mio lavoro, congiuntamente alle variabili, le tecniche, gli operatori e i parametri del modello di apprendimento per ottenerli, consiglio la visione del notebook best\_result.ipynb.
## Materiale extra
Alcune slide di presentazione del progetto sono disponibili [qui](https://docs.google.com/presentation/d/1GH-OsCUFrqLLk-CFYR8U-zfByxMaecM4rlz58dKerbE/edit?usp=sharing).
