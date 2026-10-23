# Résolution d'équations différentielles par la méthode de Galerkin

Projet de M1 : implémentation en Python d'un solveur pour l'équation de Poisson 2D basé sur la **méthode de Galerkin** (éléments finis 1D en tensorisation), pour des problèmes de type **Neumann** ou **Dirichlet**.

Le programme calcule la solution numérique sur des maillages de plus en plus fins, la compare à une solution exacte connue, puis estime **l'ordre de convergence** de la méthode.

## Principe de la méthode

Le problème est présenté de manière plus précise dans un pdf que je mettrai ou non (dépendant de l'accord de mes professeurs).

Le problème est résolu sur un domaine carré, avec un maillage uniforme en `x` et en `y` (le maillage peut toutefois devenir rectangulaire avec certaines modifications). Grâce à la structure tensorielle, le système 2D se ramène à deux problèmes 1D :

1. Construction des matrices de rigidité `Ax`, `Ay` et de masse `Bx`, `By`, assemblées à partir des matrices élémentaires :
   - `a_hat = [[1, -1], [-1, 1]]`
   - `b_hat = [[1/3, 1/6], [1/6, 1/3]]`
2. Calcul de `Ax_tilde = Bx⁻¹ · Ax` et `Ay_tilde = By⁻¹ · Ay`.
3. Diagonalisation : `Ax_tilde = P · Λx · P⁻¹` et `Ay_tilde = Q · Λy · Q⁻¹`.
4. Résolution dans la base des vecteurs propres :
   `Û[i, j] = F̂[i, j] / (λx[i] + λy[j])`
   (les valeurs propres quasi nulles, `< 1e-7`, sont ignorées : c'est typique du cas Neumann où le noyau contient les constantes).
5. Retour à la base canonique : `U = P · Û · Qᵀ`.

### Cas test

| Élément | Expression |
|---|---|
| Solution exacte | `u(x, y) = cos(x) · cos(y)` |
| Second membre | `f(x, y) = 2 · cos(x) · cos(y)` |

L'erreur est mesurée en norme infinie : `max |U - V|`, où `V` est la solution exacte évaluée sur le maillage.

## Structure du projet

Le projet se décline en 2 dossiers qui ont la même structure. Un pour le cas Neumann et un pour le cas Dirichlet.

```
diff_equation_solver/Neumann/
├── main.py                 # Script principal : étude de convergence + graphiques
├── mesh_generation.py      # Génération du maillage (X, Y) pour un nombre de points donné
├── matrix_creation.py      # Matrices élémentaires a_hat et b_hat
├── matrix_filler.py        # Assemblage des matrices Ax, Ay, Bx, By (version 1, versio boucle)
├── matrix_filler_bis.py    # Assemblage des matrices Ax, Ay, Bx, By (version utilisée par main.py)
├── utils.py                # Inversion, diagonalisation, solution exacte, second membre
├── requirements.txt        # Dépendances Python
├── .flake8                 # Configuration du linter
└── results/                # Résultats / figures produits
```

## Installation

Pour exécuter le code, et obtenir les résultats, voici la marche à suivre

### Prérequis

- Python 3.10 ou supérieur (J'ai utilisé la version 3.14.7 de mon côté)
- `pip`

### Étapes

```bash
# 1. Cloner le dépôt
git clone https://github.com/Maelian25/diff_equation_solver.git
cd diff_equation_solver

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Lancer l'étude de convergence

2 arguments peuvent être spécifiés :
  - "-pb" qui est le nom du problème pour les noms de fichiers des résultats
  - "-s" qui spécifie si on veut sauvegarder ou non les outputs du code (si on l'omet alors on ne sauvegarde pas)

```bash
cd Dirichlet (ou cd Neumann)
python main.py -pb name_pb -s
```

Le script :

1. Boucle sur des maillages de `n_min = 10` à `n_max = 100` points, par pas de 2.
2. Pour chaque maillage, assemble les matrices, résout le problème et calcule l'erreur maximale.
3. Affiche dans le terminal l'**ordre de convergence** estimé (pente de la régression linéaire de `log(erreur)` en fonction de `log(Δx)`) et le temps de calcul total.
4. Génère et enregistre trois fichiers dans le dossier results :

| Fichier | Contenu |
|---|---|
| `log_e_fn_delta_x.pdf` | `log(erreur max)` en fonction de `log(Δx)` |
| `min_eigen_value_graph.pdf` | Évolution de la plus petite valeur propre de `Ax_tilde` et `Ay_tilde` |
| `linear_regression_plot.pdf` | Erreur mesurée et droite de régression (ordre de la méthode) |

Exemple de sortie console :

```
Convergence order : 1.99
Time to compute 90 iterations : 12.34 s
```

(valeurs données à titre d'exemple, elles dépendent de la machine utilisée)

### Modifier les paramètres

Dans `main.py`, il est tout à fait possible de changer l'étendue de l'étude :

```python
n_min = 10    # nombre minimal de points du maillage
n_max = 100   # nombre maximal de points du maillage
```

Plus `n_max` est grand, plus le calcul est long : la diagonalisation de matrices pleines coûte en O(n³).

### Résoudre un autre problème

Pour changer le cas test, modifiez dans `utils.py` :

- `exact_solution(x, y)` : la solution exacte (pour calculer l'erreur) ;
- `second_member(x, y)` : le second membre `f(x, y)` correspondant.

## Remarques

- `matrix_filler.py` et `matrix_filler_bis.py` sont deux versions de l'assemblage ; `main.py` utilise `matrix_filler_bis.py`.
- `matrix_creation.py` affiche la taille des matrices élémentaires à l'import : c'est normal.
- La diagonalisation est faite avec `numpy.linalg.eig` ; les petites parties imaginaires parasites dues aux erreurs d'arrondi sont supprimées (on garde la partie réelle).

## Auteur

Projet réalisé par [Maëlian](https://github.com/Maelian25) dans le cadre du M1 MSME à Toulouse.
