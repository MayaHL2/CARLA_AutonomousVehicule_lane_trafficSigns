# Projet d'Agent Autonome sur CARLA

Ce projet vise à développer un agent autonome de conduite sur la plateforme de simulation CARLA. L'agent est équipé de capteurs, dont des caméras, pour percevoir son environnement. Les principales réalisations sont les suivantes :

## Environnement
L'agent a été équipé de capteurs tels que des caméras et un capteur de collision. Des algorithmes de planification de chemin et de commande du véhicule ont été implémentés pour assurer une conduite autonome.

## Détection des Panneaux de Signalisation
Différents modèles de détection de panneaux de signalisation ont été évalués en utilisant des données réelles et synthétiques. Le modèle SSD-mobilenet a été choisi pour sa performance en temps réel.

## Classification des Panneaux de Signalisation
Un modèle de classification des panneaux de signalisation a été entraîné sur des données réelles. Des méthodes d'amélioration ont été explorées pour augmenter la performance avec des données synthétiques. Les résultats sont à améliorer, éventuellement, une base de données issue du simulateur permettrait d'obtenir de meilleures performances. 

## Détection des Voies de Circulation
Un outil d'annotation automatique a été développé pour annoter les images de voies de circulation générées dans CARLA. Plus de 4000 images ont été créées pour former un modèle de détection de voies.


## Prérequis et Installation

Avant de commencer, assurez-vous de disposer des éléments suivants :

- **Simulateur CARLA :** Vous pouvez télécharger et installer le simulateur CARLA en suivant les instructions fournies dans la documentation officielle : [CARLA Quick Start Guide](https://carla.readthedocs.io/en/latest/start_quickstart/). Assurez-vous de satisfaire tous les prérequis nécessaires.

- **Bibliothèques Python :** Assurez-vous d'avoir installé les bibliothèques Python suivantes via `pip` :
  ```bash
  pip install networkx tensorflow pytorch opencv-python
  ```

## Navigation et commande du véhicule  
Pour instancier un véhicule et y ajouter les capteurs, la classe `vehicle` peut être utilisée. Voici un example : 

```python
# Charger la carte "Town10HD" dans le monde
world = client.load_world('Town10HD')
map = world.get_map()

# Changer les conditions météorologiques (ClearNight, ClearNoon, etc.)
world.set_weather(carla.WeatherParameters.ClearNoon)

# Instancier un objet de la classe Vehicle
car = Vehicle(world)

# Faire apparaître le véhicule dans le monde
vehicle = car.spawn_vehicle()

# Faire apparaître une caméra et une caméra sémantique sur le véhicule
camera = car.spawn_camera(disp_size=disp_size)
sem_camera = car.spawn_semantic_camera(disp_size=disp_size)

# Faire apparaître un capteur de collision sur le véhicule
collision_sensor = car.spawn_collision_sensor()

# Démarrer l'écoute des données des capteurs
car.start_listen()

# Obtenir un objet Spectator pour voir le monde depuis une perspective extérieure
spectator = car.get_spectator()

# Obtenir les données des capteurs
sensor_data = car.get_sensor_data()

# Choisir un point aléatoire sur la carte comme point de départ
start = vehicle.get_location()

# Choisir un point d'arrivée aléatoire parmi les points d'apparition sur la carte
goal = random.choice(map.get_spawn_points()).location
```

Pour commander le véhicule, c'est la classe `navigation` qui entre en jeux. Elle permet de faire déplacer un véhicule instancié d'un point A à un point B à une vitesse désirée par le plus court chemin : 

```python 
# Instancier un objet de la classe Navigation
Nav = Navigation(map, vehicle, spectator, start, goal, desired_speed, sensor_data=sensor_data, sensors={'camera': camera, "collision_sensor": collision_sensor, "sem_camera": sem_camera})

# Lancer la navigation pendant 60 secondes avec affichage des caméras, sauvegarde de données et de vidéo
# Les données images sont affichées toutes les 0.05s
# plusieurs paramètres peuvent être modifiés, lire les commentaires de la classe pour plus de détails.
Nav.run_navigation(display_cameras=True, save_data=True, camera_step_save=0.05, save_video=True, time_limit=60)

# Dessiner la route sur la carte
Nav.draw_route()

# Dessiner le graphique d'erreur de vitesse
Nav.draw_graph_error("speed_error")

# Arrêter les capteurs
car.stop_sensors()

# Détruire le véhicule
car.destroy_vehicle()

```


## Détection des panneaux de signalisation 

Pour cette étape, les travaux de [Traffic Sign Detection](https://github.com/aarcosg/traffic-sign-detection) ont été utilisés. 
Un bloc de perception appelé `TS_detection_bloc` pour la détection des panneaux en temps réel, a été ajouté à la classe `navigation`. le fichier `TS_detection` contient l'implémentation du réseau de neurones pour cette étape.


## Entraînement de la Classification des Panneaux de Signalisation

Pour l'entraînement de la classification des panneaux de signalisation, suivez ces étapes :

### Téléchargement des Bases de Données :

1. Téléchargez les bases de données GTSDB, Tsinghua-Tencent 100k (traffic signs) et EVO. Assurez-vous de découper les images de Tsinghua-Tencent 100k et EVO pour ne conserver que les panneaux de signalisation, en gardant uniquement les classes présentes dans GTSDB.
2. Utilisez le fichier CSV "EVO_to_GTSDB" pour les équivalences d'annotations entre EVO et GTSDB.
3. Téléchargez la base de données "classification_panneaux_carla" depuis le SharePoint (Cette base de données contient seulement 352 images et peut être utilisée pour les tests ou intégrée dans le jeu de données de validation).

### Modèles
Plusieurs modèles se trouvent dans le répertoire model_TrafficSignNet et peuvent être testés ou utilisés comme modèles pré-entraînés.


## Détection des voies de circulation

Dans cette section, nous avons travaillé sur la création d'une base de données annotée en utilisant des images générées par le simulateur CARLA. Le fichier `lane_detection_annotation` contient des fonctions spécifiquement conçues pour réaliser l'annotation et la classification des voies de circulation. Ces fonctions ont ensuite été appelées dans la classe `navigation` (via la fonction `lane_annotation`) pour permettre l'annotation en temps réel.

Une fois la base de données créée, il est possible de visualiser les annotations en utilisant le code présent dans le fichier `visualize_annotation_lane`.

Étant donné que le simulateur n'affiche que des images de taille $2^n$, le fichier `lane_resize` a été conçu pour supprimer 164 pixels du haut et du bas de l'image. Cela permet aux images d'être compatibles avec l'entraînement de [CondLaneNet](https://github.com/aliyun/conditional-lane-detection/blob/master/docs/install.md). Les annotations sont également modifiées en conséquence.

Le fichier `path_reader` est utilisé pour créer la structure de l'arborescence de la base de données. Cela est nécessaire lors de l'entraînement avec [CondLaneNet](https://github.com/aliyun/conditional-lane-detection/blob/master/docs/install.md).
