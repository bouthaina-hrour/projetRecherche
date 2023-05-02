# Projet Recherche 
* Les videos utilisés dans cette étude sont placées dans le dossier **DataSet**

## Partie 1 : approche OpenPose 
* Cette approche est implémentée dans le dossier **tf-pose-estimation** .
* Le fichier ``setup.py`` permet l'installation des packages tf_pose nécessaires à son éxecution.


* Les fichiers ``run_webcam.py`` et ``run_video.py`` capturent les images à partir des videos pour estimer la posture du joueur en enregistrant les résultats sous format vidéo et frames.


* Les résultats obtenus sont dans le dossier **VideosResultsTfpose**. 

## Partie 2 : approche Blender 
* L'approche consiste à utiliser le logiciel Blender pour l'estimation de la posture du joueur lors des lancers francs réussi et échoué. 

* Les projets Blender (dont l'extension est ``.blend``) se trouvent au dossier nommé **Blender** ainsi que les scripts utilisés pour l'extraction des positions des objets blender correspondants aux articulations. 


* Les scripts utilisés pour l'estimation sont dans le dossier **Blender/blender_scripts**. 

* Les résultats obtenus par les projets blender sont sous format ``.csv`` et sont placés dans le dossier **Blender/Results**

* Le dossier **graphBadAndGoodShoot** contient les comparaisons entre les positions des deux lancers et les scripts associés
* Le dossier **correction** contient les scripts utilisés pour la correction des lancers francs échoués ainsi que les résultats obtenus. 




