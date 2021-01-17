# Projet Application - Transfert de style

![Supported Python Versions](https://img.shields.io/badge/Python->=3.8-blue.svg?logo=python&logoColor=white) ![Made withJupyter](https://img.shields.io/badge/Jupyter-6.1.5-orange.svg?logo=jupyter&logoColor=white)

_Auteurs:_ [Simon Audrix](mailto:saudrix@ensc.fr),  [Tarek Lammouchi](mailto:tarek.lammouchi@bordeaux-inp.fr), [Quentin Lanneau](mailto:quentin.lanneau@bordeaux-inp.fr), [Gabriel Nativel-Fontaine](mailto:gnativ910e@ensc.fr) 

Le rapport du projet dans le fichier [rapport.pdf](https://github.com/Gab1i/vision-style-transfer/blob/main/rapport.pdf)

## Arborescence

- Le dossier notebooks contient des notebooks qui nous ont permis de faire quelques tests et d'obtenir certaines des images du rapport.
  - Gaty's algorithm.ipynb contient les différents tests que nous avons effectués sur l'algorithme de Gaty
  - Neural_Style_Transfer_Demo.ipynb est une version préliminaire de démonstration de l'algorithme de Gaty
  - Unsupervised Transfer Style.ipynb est une version non fonctionnelle d'un algorithme que nous trouvions intéressant à implémanter mais que nous n'avons pas eu le temps de terminer.
- Le dossier src contient les sources en python que nous avons développé tout au long du projet (certains codes développé dans ces fichiers ne sont pas fonctionnels)
  - Cycle_gan_full.py est une version issue du repository https://github.com/keras-team/keras-io/blob/master/examples/generative/. C'est celle qui a été utilisée pour les tests
  - Cycle_gan_perso.py est la version que nous avons tenté de développer. Elle souffre de problèmes, mais nous n'avons pas eu le temps de déterminer lesquels. Nous avons préféré effectuer nos tests sur d'autres implémentations
  - CycleGanModel.py, est une version issue de [Cycle Gan Tutorial with Keras](https://machinelearningmastery.com/cyclegan-tutorial-with-keras/) modifiée en tant que classe héritant de la class Model de tensorflow.keras
  - StyleTransferModel.py, est une version issue de [mlhandbook](https://github.com/bpesquet/mlhandbook) modifiée en tant que classe héritant de la class Model de tensorflow.keras
  - Utils.py contient un ensemble de fonctions utiles