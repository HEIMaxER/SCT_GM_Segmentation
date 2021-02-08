Ce code s'inscrit dans le cadre d'une étude de la segmentation de la moelle épinière sur le jeu de données "Spinal cord grey matter segmentation challenge" (PRADOS, Ferran, ASHBURNER, John, BLAIOTTA, Claudia, et al. Spinal cord grey matter segmentation challenge. Neuroimage, 2017, vol. 152, p. 312-329.).

Il utilise Spinal Corde Toolbox (SCT) v5.1.0 qui est un logiciel open-source de traitement d'images d'IRM. Pour plus d'informations visiter : https://spinalcordtoolbox.com/en/stable/

### Installation

1. Emplacement

Le dossier `Code_Segmentation_SCT` contenant ce code a été conçu pour être dans même emplacement que le dossier `images_db` contenant les IRMs.

2. Installation de SCT
 
Le programme se basant sur la fonction de segmentation de SCT il faut dans un premier temps installer SCT en suivant les étapes décrites sur la page web suivante :

- https://spinalcordtoolbox.com/en/stable/user_section/installation.html

A noter que le programme utilise la version 5.1.0 donc après avoir fait une copie du dépôt github il faut sélectionner la branche 5.1.0 :

 ```Shell
git checkout 5.1.0
 ```

Une fois SCT installé, assurez vous d'utiliser l'environnement virtuel anaconda installé :

````Shell
source ${SCT_DIR}/python/etc/profile.d/conda.sh
conda activate venv_sct
````

3. Installation pour le notebook jupyter

Afin de pouvoir exécuter le code du notebook avec l'environnement virtuel SCT il faut installer jupyter ainsi que ipykernel et ajouter l'environnement SCT :

 ```Shell
conda install jupyter

conda install -c anaconda ipykernel

python -m ipykernel install --user --name=venv_sct
 ```

### Execution

L'ensemble des commandes ont été prévues pour être exécutées dans le dossier `Code_Segmentation_SCT` dans l'environnement virtuel `venv_sct` installé par SCT.

1. Notebook

La majorité du travail se trouve dans le notebook `Analyse_Segmentation_Moelle_Epiniere.ipyndb`. Il est prêt à être exécuté une fois l'installation terminée, assurez vous que jupyter utilise bien l'environement `venv_sct`

2. Commandes

Plusieurs scriptes ont été utilisés pour obtenir les résultats exposés dans le notebook. Premièrement la segmentation réalisées à l'aide d'SCT :

- Commande générant l'ensembles des masques SCT pour les données d'entrainement et les sauvegardant dans `/images_db/SCT/training_masks/` :
```Shell
python run_sct_seg_on_folder.py -i ../images_db/training-data-gm-sc-challenge-ismrm16-v20160302b/ -o ../images_db/SCT/training_masks/
```

- Commande générant l'ensembles des masques SCT pour les données test et les sauvegardant dans `/images_db/SCT/test_masks/` :
```Shell
python run_sct_seg_on_folder.py -i ../images_db/test-data-gm-sc-challenge-ismrm16-v20160401/ -o ../images_db/SCT/test_masks/
```

Ensuite les scores mesurant la performances de SCT :

- Commande calculant les scores Dice et les distances de Hausdorff des masques SCT :
```Shell
python get_sct_seg_stats.py -gt ../images_db/training-data-gm-sc-challenge-ismrm16-v20160302b/ -sct ../images_db/SCT/training_masks/ -o stat_files/
```

- Commande calculant les scores Dice et les distances de Hausdorff entre les masques experts :
```Shell
python get_mask_cross_scores.py -gt ../images_db/training-data-gm-sc-challenge-ismrm16-v20160302b/ -o stat_files/
```

Enfin les commandes visant à analyser la moelle épinière :

- Commande générant les squelette de référence pour les l'analyse de forme :
```Shell
python make_gm_skeleton_template.py -gt ../images_db/training-data-gm-sc-challenge-ismrm16-v20160302b/ -o stat_files/
```

- Commande qui extrait les propriétés de la moelle épinières des données training :
```Shell
python get_GM_features.py -gt ../images_db/training-data-gm-sc-challenge-ismrm16-v20160302b/ -sct ../images_db/SCT/training_masks/ -mt stat_files/average_masks.pkl -o stat_files/
```

- Commande qui extrait les propriétés de la moelle épinières des données test :
```Shell
python get_GM_features.py -gt ../images_db/test-data-gm-sc-challenge-ismrm16-v20160401/ -sct ../images_db/SCT/test_masks/ -mt stat_files/average_masks.pkl -o stat_files/
```
3. Commandes SCT

L'ensemble du code n'utilise qu'une commande SCT servant à réaliser la segmentation de la moelle épinière :

```Shell
sct_deepseg_gm -i -o
```

Cependant SCT possède de nombreux autres outils d'analyse. Pour plus d'informations : https://spinalcordtoolbox.com/en/stable/user_section/command-line.html


