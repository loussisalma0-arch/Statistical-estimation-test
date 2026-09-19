# Statistical Estimation Test — Impact de BADR sur la performance logistique (Groupe Auto Hall)

Code Python et données d'enquête utilisés dans le cadre d'un Projet de Fin d'Études (PFE) portant sur l'impact du système douanier électronique **BADR** (Base Automatisée des Douanes en Réseau, Maroc) sur la **performance logistique** perçue, à partir d'une étude de cas menée auprès du **Groupe Auto Hall**.

> Mémoire : *La digitalisation comme levier de performance dans la logistique du commerce international — Étude de l'impact du système douanier BADR sur la performance logistique, cas du Groupe Auto Hall*
> Université Mohammed V de Rabat — Master Parcours d'Excellence en Relations Commerciales Internationales
> Réalisé par **Salma Loussi** — Encadrant : **Pr. Driss Mafamane**

## Contenu du dépôt

| Fichier | Description |
|---|---|
| `analyse_estimation_BADR.py` | Script principal : nettoyage des données, calcul des scores composites, fiabilité (alpha de Cronbach), statistiques descriptives, corrélations, régression hiérarchique, diagnostics, modération, médiation, synthèse des hypothèses. |
| `reponses_badr.csv` | Données brutes exportées du questionnaire (Google Forms), anonymisées — 31 réponses collectées. |
| `donnees_composites.csv` | Fichier généré par le script : données brutes + variables composites calculées (TAM, TOE, RBV, PERF, etc.), utilisé pour les analyses et les visualisations. |

## Contexte de la recherche

Le modèle testé s'appuie sur un cadre théorique intégrant trois grilles de lecture de l'adoption technologique :

- **TAM** (*Technology Acceptance Model*, Davis, 1989) — utilité perçue et facilité d'usage.
- **TOE** (*Technology-Organization-Environment*, Tornatzky & Fleischer, 1990) — ressources internes et soutien de la direction.
- **RBV** (*Resource-Based View*, Barney, 1991) — formation, intégration aux systèmes internes, procédures.

Ces trois construits sont mis en relation avec la performance logistique perçue (**PERF**) et ses composantes (délais, coûts, fiabilité, traçabilité), avec l'ancienneté d'usage de BADR (**ANC**) comme variable de contrôle et variable modératrice.

Quatre hypothèses sont testées :

1. **H1** — TAM améliore la performance logistique perçue.
2. **H2** — TOE améliore la performance logistique perçue.
3. **H3** — RBV améliore la performance logistique perçue.
4. **H4** — L'ancienneté d'usage (ANC) modère l'effet de TAM sur la performance perçue.

## Description des données

`reponses_badr.csv` contient les réponses brutes au questionnaire (échelle de Likert 1 à 5, sauf indication contraire). Les colonnes couvrent :

- **Profil du répondant** : service, fonction, ancienneté au sein du groupe, ancienneté d'usage de BADR, fréquence d'usage, formation reçue.
- **Usage et maîtrise de BADR** : items d'utilité perçue, de facilité d'usage, de ressources internes, de soutien de la direction, de cadre réglementaire/formation, d'intégration aux systèmes internes, de procédures.
- **Impact perçu** : délais logistiques, coûts, fiabilité, traçabilité.
- **Évaluation globale** : performance logistique perçue, remarque ouverte facultative.

`donnees_composites.csv` ajoute les variables composites calculées par le script (moyenne des items bruts par construit) : `UP`, `FU`, `TAM`, `RI`, `SD`, `TOE`, `FOR`, `INT`, `PROC`, `RBV`, `DELAI`, `COUT`, `FIAB`, `TRACA`, `PERF`, `ANC`.

> **Remarque méthodologique** : dans le questionnaire effectivement diffusé, la section libellée « Cadre réglementaire » a par erreur soumis deux fois les items de la rubrique « Formation ». Le construit `TOE` exploité dans les analyses repose donc uniquement sur les ressources internes et le soutien de la direction (items de cadre réglementaire non disponibles). Cette anomalie est documentée en détail dans le chapitre 3 du mémoire.

L'échantillon brut compte 31 réponses ; 2 répondants ayant indiqué ne jamais utiliser BADR sont exclus des analyses statistiques, portant l'échantillon analytique à **n = 29**.

## Prérequis

- Python ≥ 3.9
- Bibliothèques : `pandas`, `numpy`, `scipy`, `statsmodels`, `pingouin`

Installation rapide :

```bash
pip install pandas numpy scipy statsmodels pingouin
```

Ou, si un fichier `requirements.txt` est ajouté au dépôt :

```bash
pip install -r requirements.txt
```

## Utilisation

Depuis un terminal, dans le dossier du dépôt :

```bash
python analyse_estimation_BADR.py
```

Le script :

1. lit `reponses_badr.csv` (doit se trouver dans le même dossier que le script) ;
2. calcule les scores composites et les affiche avec les statistiques descriptives et les corrélations ;
3. estime le modèle de régression hiérarchique (3 blocs), les diagnostics (VIF, Shapiro-Wilk, Breusch-Pagan, Durbin-Watson), le test de modération et les analyses de médiation ;
4. affiche la décision (validée / non confirmée) pour chacune des quatre hypothèses ;
5. écrit le fichier `donnees_composites.csv` en sortie, dans le même dossier.

## Méthodes statistiques appliquées

1. **Fiabilité des échelles** — Alpha de Cronbach par construit (seuil d'acceptabilité α ≥ 0,70).
2. **Statistiques descriptives et corrélations** — moyennes, écarts-types, corrélations de Pearson et de Spearman.
3. **Régression linéaire multiple hiérarchique (MCO)** — modèle en 3 blocs successifs (ANC, puis + TAM, puis + TOE + RBV).
4. **Diagnostics du modèle** — VIF (multicolinéarité), Shapiro-Wilk (normalité des résidus), Breusch-Pagan (homoscédasticité), Durbin-Watson (indépendance des résidus).
5. **Test de modération** — régression avec terme d'interaction sur variables centrées (test de H4).
6. **Analyse de médiation** — effets directs et indirects par bootstrap (5000 réplications, IC à 95 %), pour la chaîne causale TOE → TAM → RBV → Performance.
7. **Synthèse des hypothèses** — décision automatique (validée / non confirmée / effet inverse) à partir des p-values et des signes des coefficients.

## Résultats principaux (résumé)

| Hypothèse | Résultat |
|---|---|
| H1 (TAM → performance) | **Validée** (β = 0,701 ; p = 0,024) |
| H2 (TOE → performance) | Non confirmée en régression multiple (effet positif en corrélation bivariée ; médiation totale via TAM) |
| H3 (RBV → performance) | **Validée** (β = 0,605 ; p = 0,023) |
| H4 (modération par l'ancienneté) | Non confirmée (p = 0,184 ; puissance statistique probablement insuffisante, n = 29) |

Modèle final : R² = 0,738, R² ajusté = 0,694, F = 16,90 (p < 0,001).

Le détail complet des résultats, leur discussion et leur mise en perspective théorique et institutionnelle sont présentés dans le chapitre 3 du mémoire (non inclus dans ce dépôt).

## Confidentialité et éthique

Les données ont été collectées de manière anonyme (aucune donnée nominative) auprès de collaborateurs du Groupe Auto Hall, dans le cadre strict d'un travail académique. Elles sont exploitées ici à des fins de reproductibilité de l'analyse statistique du mémoire.

## Citation suggérée

> Loussi, S. (2026). *La digitalisation comme levier de performance dans la logistique du commerce international — Étude de l'impact du système douanier BADR sur la performance logistique, cas du Groupe Auto Hall.* Projet de Fin d'Études, Master Parcours d'Excellence en Relations Commerciales Internationales, Université Mohammed V de Rabat.

## Auteur

**Salma Loussi** — Master Parcours d'Excellence en Relations Commerciales Internationales, Université Mohammed V de Rabat.
