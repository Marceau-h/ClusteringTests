# Les corpus

## 3 langues
* corpus --> corpus français
* corpus_en --> corpus anglais
* corpus_pt --> corpus portugais

## Architecture dossier

* Un dossier par auteur ; 
* Pour chaque auteur : 
 * un dossier REF avec le texte de référence et les clusters d'entités nommées de lieu ;
 * un dossier OCR : 1 dossier par OCR. Dans chacun le texte OCR et les cluster d'entités nommées de lieu ; 
 
# Cluster

Nous avons utilisé plusieurs algorithme de clustering dont les noms figurent dans le noms de fichiers :
 
* AffpropDistA1_clusters
* AffpropHyperparams2_clusters
* AffpropHyperparams_clusters
* AffpropKeepVectors_clusters
* DBScan_clusters
* Default_clusters
* HDBScan_clusters
* KMeans_clusters
* Optics_clusters.

Les fichiers qui se terminent par :
* *cluster.json comportent un dictionnaire des clusters, dont les clés sont le numéro du cluster et la valeur la liste des termes du cluster ;
* *corr.csv et * *cor.jsonl sont des fichiers pour l'annotation en vue de l'évaluation des clusters.

## Notes à propos des algorithmes de clustering utilisés :

* Affinité propagation : Affprop default :
 * le centroïde dans Affprop est calculé en réduisant la dimension a deux et calcul de la moyenne de tous les x et de tous les y ;
 * test de différents paramétrages ;
* hyperparam : changer les hyperparamètres ;
* DistA1 : distance a 1 si moins de 2 ngral en communs ;
* KeepVectors : dès qu’il y a moins de 1 3grammes en commun il vont dans cluster-1 ;
* SbBscan : plus proche voisin agglomératif ;

### Ce qui nous a convaincu :
* Affpropohyperparams2 :  Dans le CountVectorizer paramétrage du min_df = 3 pour que seul les bigrammes et trigrammes présent dans plus 2 tokens –> tous les tokens avec 0 colonnes -> hors clustering 🙂✅ 🎉


### Finalement nous avons laissé de côté:
 * Dendrogramme Hierarchical Bisecting K-means, il faut donner un nombre de cluster et du coup tout été dans le même cluster. Il a tourné sur un K de 2 à 20 et conservation du meilleur silhouette-score  ❌ 
 * OPTICS → utiliser une distance cosinus
 * DBscan → utiliser une distance cosinus
 * compter le nb de bigramme et envoyer à l’algo si $bigramme < 2$ -> mettre la distance à 1 ❌

# Annotation manuelle pour l'évaluation du clustering :

## Fichiers annotés
Les fichiers dont le nom se termine par corr-annot-OK.csv sont ré-annotés manuellement pour l'évaluation du clustering.

* Dans le dossier corpus : AIMARD-TRAPPEURS ;
* Dans corpus\_en : AGUILAR\_home-influence ;

## Description du fichier annoté
* La colonne text comporte les mots du cluster ligne par ligne ;
* La colonne cluster comporte le numéro du cluster attribué lors du calcul des clusters ;
* La colonne cluster_corrected comporte le numéro du cluster ;

* Retirer des clusters quand dans la cellule il n’y a que  : 
 * des nombres/chiffres, mettre dans le cluster -1
 * suite de bi-tri-grammes de caractère qui se suivent plus de 2*, -1

## Consignes pour l'annotation des clusters

* des ponctuations qui se suivent : …,,.., poubelle
 * plusieurs fois le même caractère unique  qui se suit (ccc), \w{2,} poubelle
 * les mots de 1 caractères paramétrage utilisateur
 * un mélange de minucules, majuscules, nombre, chiffres, ponctuations


* Signes annotations : 

 * “ ?” quand je ne vois pas pourquoi le terme fait parti du cluster mais que j’ai un doute
 * “‘  “ quand le mot ne devrait pas faire partie du cluster selon moi
 * “ $”  indique un problème dans l ‘annotation du csv origine
 * -1 quand c’est pas une entité et qu’il est tout seul



limite de l’affinité de propagation : des mots sont mis dans le même cluster  alors qu’ils ne semblent pas avoir de bi-tri-gr commun, c’est parcequ’ils ont un bi-tri-gr commun avec un troisième




