"""
================================================================================
 ANALYSE EMPIRIQUE — Impact du système BADR sur la performance logistique
 Cas du Groupe Auto Hall — PFE Salma Loussi (Master RCI, UM5 Rabat)
================================================================================

Ce script reproduit l'ensemble des estimations présentées au chapitre 3 :
  1. Fiabilité des échelles (Alpha de Cronbach)
  2. Statistiques descriptives des construits composites
  3. Analyse corrélationnelle (Pearson + Spearman)
  4. Régression linéaire multiple hiérarchique (modèle principal, H1-H3)
  5. Diagnostics du modèle (VIF, Shapiro-Wilk, Breusch-Pagan, Durbin-Watson)
  6. Test de modération (H4 : ancienneté x TAM)
  7. Analyse de médiation en chaîne (TOE -> TAM -> PERF ; TAM -> RBV -> PERF)
  8. Synthèse de la décision sur les hypothèses H1 à H4

--------------------------------------------------------------------------------
DONNÉES D'ENTRÉE
--------------------------------------------------------------------------------
Le script attend un fichier CSV nommé "reponses_badr.csv" dans le même dossier,
au format exporté par Google Forms (Réponses -> icône verte Sheets -> Fichier ->
Télécharger -> Valeurs séparées par une virgule .csv), avec les colonnes dans
le même ordre que le questionnaire (annexe 1) : profil (colonnes 1 à 6), puis
les items de mesure (colonnes 7 à 38), puis la remarque ouverte (colonne 39).

--------------------------------------------------------------------------------
INSTALLATION
--------------------------------------------------------------------------------
    pip install pandas numpy scipy statsmodels pingouin

--------------------------------------------------------------------------------
EXÉCUTION
--------------------------------------------------------------------------------
    python analyse_estimation_BADR.py

Le script affiche tous les résultats dans la console (à copier/capturer en
"screenshots" pour le mémoire ou la soutenance) et écrit en sortie le fichier
"donnees_composites.csv" contenant l'ensemble des variables composites
calculées, réutilisable pour des analyses complémentaires.

NOTE : la fenêtre reste ouverte à la fin (ou en cas d'erreur) et attend que
vous appuyiez sur Entrée avant de se fermer, pour que vous ayez le temps de
lire les résultats même en double-cliquant sur le fichier.
================================================================================
"""

import os
import sys
import traceback


def main():
    import pandas as pd
    import numpy as np
    import pingouin as pg
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.diagnostic import het_breuschpagan
    from scipy import stats

    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)

    # On cherche les fichiers à côté du script lui-même, pas dans le dossier
    # "courant" (qui, sur Windows, est parfois le dossier de Python quand on
    # double-clique sur le .py au lieu de le lancer depuis l'invite de commandes).
    try:
        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        # __file__ n'existe pas quand ce code est collé directement dans la
        # console Python (>>>) au lieu d'être exécuté comme fichier .py.
        print("=" * 78)
        print("ATTENTION : ce code semble avoir été collé directement dans la")
        print("console Python (>>>), et non exécuté comme un fichier .py.")
        print("Fermez cette console (tapez exit()) et lancez plutôt, depuis")
        print("l'invite de commandes (cmd), la ligne :")
        print("    python analyse_estimation_BADR.py")
        print("=" * 78)
        SCRIPT_DIR = os.getcwd()
    CSV_PATH = os.path.join(SCRIPT_DIR, "reponses_badr.csv")
    OUT_PATH = os.path.join(SCRIPT_DIR, "donnees_composites.csv")

    # ==========================================================================
    # 0. CHARGEMENT ET PRÉPARATION DES DONNÉES
    # ==========================================================================

    if not os.path.exists(CSV_PATH):
        print("=" * 78)
        print(f"ERREUR : le fichier '{CSV_PATH}' est introuvable.")
        print(f"Dossier dans lequel le script cherche ce fichier : {os.getcwd()}")
        print()
        print("Solutions :")
        print(f"  1) Vérifiez que '{CSV_PATH}' est bien copié dans CE dossier "
              f"(le même que le script .py).")
        print("  2) Si votre fichier a un autre nom (ex. 'Réponses.csv'), soit vous le "
              f"renommez en '{CSV_PATH}', soit vous changez la valeur de CSV_PATH "
              "en haut de cette fonction.")
        print("  3) Pour obtenir ce fichier : ouvrez votre Google Form -> onglet "
              "Réponses -> icône verte Sheets -> dans Google Sheets : "
              "Fichier > Télécharger > Valeurs séparées par une virgule (.csv).")
        print("=" * 78)
        return

    df_brut = pd.read_csv(CSV_PATH)
    cols = list(df_brut.columns)

    # ---- Dictionnaire de correspondance colonne CSV <-> code interne ----
    # (indices positionnels confirmés sur l'export Google Forms du questionnaire, annexe 1)
    c = {
        "service": cols[1], "fonction": cols[2], "anc_groupe": cols[3],
        "anc_badr": cols[4], "freq": cols[5], "formation_oui_non": cols[6],
        "up1": cols[7], "up2": cols[8], "up3": cols[9],                      # Utilité perçue (TAM)
        "fu1": cols[10], "fu2": cols[11], "fu3": cols[12],                   # Facilité d'usage (TAM)
        "ri1": cols[13], "ri2": cols[14],                                    # Ressources internes (TOE)
        "sd1": cols[15], "sd2": cols[16],                                    # Soutien direction (TOE)
        "for1": cols[17], "for2": cols[18],                                  # Formation (RBV)
        # NB : ces 2 items sont mal étiquetés "Cadre réglementaire" dans le Google
        # Form administré ; leur contenu réel est celui des items "Formation" du
        # questionnaire (annexe 1, 2.13-2.14). Les vrais items de cadre réglementaire
        # (2.11-2.12) n'ont pas été soumis -> TOE = Ressources internes + Soutien
        # direction uniquement (cf. remarque méthodologique, section 3.2.2).
        "int1": cols[19], "int2": cols[20],                                  # Intégration SI (RBV)
        "proc1": cols[21], "proc2": cols[22],                                # Procédures internes (RBV)
        "del1": cols[23], "del2": cols[24], "del3": cols[25],
        "del4": cols[26], "del5": cols[27],                                  # Délais logistiques (KPI)
        "cout1": cols[28], "cout2": cols[29],                                # Coûts (KPI)
        "fia1": cols[30], "fia2": cols[31], "fia3": cols[32],                # Fiabilité (KPI)
        "tra1": cols[33], "tra2": cols[34],                                  # Traçabilité (KPI)
        "perf1": cols[35], "perf2": cols[36], "perf3": cols[37], "perf4": cols[38],  # Performance globale (VD)
    }

    # Conversion numérique de tous les items de mesure (les colonnes de profil restent du texte)
    colonnes_profil = ("service", "fonction", "anc_groupe", "anc_badr", "freq", "formation_oui_non")
    for k, v in c.items():
        if k not in colonnes_profil:
            df_brut[v] = pd.to_numeric(df_brut[v], errors="coerce")

    # ---- Exclusion des non-utilisateurs de BADR (question filtre 1.5) ----
    n_brut = len(df_brut)
    df = df_brut[df_brut[c["freq"]] != "Jamais"].copy().reset_index(drop=True)
    n = len(df)
    print(f"=== Échantillon collecté : {n_brut} réponses | Échantillon analysé (hors non-utilisateurs) : n = {n} ===\n")

    def comp(*keys):
        """Score composite = moyenne des items Likert du construit."""
        return df[[c[k] for k in keys]].mean(axis=1)

    # ---- Calcul des variables composites (moyenne des items, échelle 1 à 5) ----
    df["UP"] = comp("up1", "up2", "up3")
    df["FU"] = comp("fu1", "fu2", "fu3")
    df["RI"] = comp("ri1", "ri2")
    df["SD"] = comp("sd1", "sd2")
    df["FOR"] = comp("for1", "for2")
    df["INT"] = comp("int1", "int2")
    df["PROC"] = comp("proc1", "proc2")
    df["DELAI"] = comp("del1", "del2", "del3", "del4", "del5")
    df["COUT"] = comp("cout1", "cout2")
    df["FIAB"] = comp("fia1", "fia2", "fia3")
    df["TRACA"] = comp("tra1", "tra2")
    df["PERF"] = comp("perf1", "perf2", "perf3", "perf4")

    # ---- Construits de second ordre du modèle conceptuel (tableau 1.7) ----
    df["TAM"] = df[["UP", "FU"]].mean(axis=1)
    df["TOE"] = df[["RI", "SD"]].mean(axis=1)
    df["RBV"] = df[["FOR", "INT", "PROC"]].mean(axis=1)

    # ---- Ancienneté d'usage de BADR : recodage numérique par valeur médiane de tranche ----
    anc_map = {"Moins d'1 an": 0.5, "1 à 3 ans": 2, "3 à 5 ans": 4, "Plus de 5 ans": 6}
    df["ANC"] = df[c["anc_badr"]].map(anc_map)

    print("--- Répartition du profil (échantillon analysé) ---")
    print(df[c["anc_badr"]].value_counts(), "\n")
    print(df[c["formation_oui_non"]].value_counts(), "\n")

    # ==========================================================================
    # 1. FIABILITÉ DES ÉCHELLES (Alpha de Cronbach)
    # ==========================================================================
    print("\n=== 1. FIABILITÉ DES ÉCHELLES (Alpha de Cronbach) ===")
    alpha_sets = {
        "TAM (UP+FU, 6 items)": ["up1", "up2", "up3", "fu1", "fu2", "fu3"],
        "TOE (RI+SD, 4 items)": ["ri1", "ri2", "sd1", "sd2"],
        "RBV (FOR+INT+PROC, 6 items)": ["for1", "for2", "int1", "int2", "proc1", "proc2"],
        "DELAI (5 items)": ["del1", "del2", "del3", "del4", "del5"],
        "COUT (2 items)": ["cout1", "cout2"],
        "FIAB (3 items)": ["fia1", "fia2", "fia3"],
        "TRACA (2 items)": ["tra1", "tra2"],
        "PERF (4 items)": ["perf1", "perf2", "perf3", "perf4"],
    }
    for label, keys in alpha_sets.items():
        sub = df[[c[k] for k in keys]]
        a, ci = pg.cronbach_alpha(data=sub)
        flag = "OK" if a >= 0.70 else ("limite" if a >= 0.60 else "FAIBLE")
        print(f"{label:32s} alpha = {a:.3f}  (IC95%: {ci})  [{flag}]")

    # ==========================================================================
    # 2. STATISTIQUES DESCRIPTIVES DES COMPOSITES
    # ==========================================================================
    print("\n=== 2. STATISTIQUES DESCRIPTIVES DES COMPOSITES ===")
    desc = df[["UP", "FU", "TAM", "RI", "SD", "TOE", "FOR", "INT", "PROC", "RBV",
               "DELAI", "COUT", "FIAB", "TRACA", "PERF", "ANC"]].describe().T[["mean", "std", "min", "max"]]
    print(desc.round(2))

    # ==========================================================================
    # 3. ANALYSE CORRÉLATIONNELLE
    # ==========================================================================
    print("\n=== 3. MATRICE DE CORRÉLATION DE PEARSON ===")
    corr_vars = ["TAM", "TOE", "RBV", "ANC", "DELAI", "COUT", "FIAB", "TRACA", "PERF"]
    corr = df[corr_vars].corr(method="pearson")
    print(corr.round(2))

    print("\n--- Corrélations de Spearman avec PERF (robustesse, données ordinales) ---")
    corr_sp = df[corr_vars].corr(method="spearman")
    print(corr_sp.round(2)["PERF"])

    # ==========================================================================
    # 4. RÉGRESSION LINÉAIRE MULTIPLE HIÉRARCHIQUE (modèle principal, VD = PERF)
    #    Teste H1 (TAM), H2 (TOE) et H3 (RBV)
    # ==========================================================================
    print("\n=== 4. RÉGRESSION LINÉAIRE MULTIPLE HIÉRARCHIQUE (VD = PERF) ===")

    m1 = smf.ols("PERF ~ ANC", data=df).fit()
    m2 = smf.ols("PERF ~ ANC + TAM", data=df).fit()
    m3 = smf.ols("PERF ~ ANC + TAM + TOE + RBV", data=df).fit()

    for name, m in [("Modèle 1 (ANC)", m1), ("Modèle 2 (+ TAM)", m2), ("Modèle 3 (+ TOE + RBV)", m3)]:
        print(f"\n--- {name} ---  R²={m.rsquared:.3f}  R²ajusté={m.rsquared_adj:.3f}  "
              f"F={m.fvalue:.2f} (p={m.f_pvalue:.4f})")
        print(m.params.round(3))
        print("p-values:", m.pvalues.round(4).to_dict())

    print(f"\nΔR² (Modèle 1 -> 2, effet TAM)      = {m2.rsquared - m1.rsquared:.3f}")
    print(f"ΔR² (Modèle 2 -> 3, effet TOE + RBV) = {m3.rsquared - m2.rsquared:.3f}")

    # ==========================================================================
    # 5. DIAGNOSTICS DU MODÈLE FINAL (modèle 3)
    # ==========================================================================
    print("\n=== 5. DIAGNOSTICS DU MODÈLE FINAL (modèle 3) ===")
    X = sm.add_constant(df[["ANC", "TAM", "TOE", "RBV"]])

    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print("VIF (multicolinéarité, seuil critique = 10) :\n", vif.round(2))

    resid = m3.resid
    sh_stat, sh_p = stats.shapiro(resid)
    print(f"\nShapiro-Wilk (normalité des résidus)   : W={sh_stat:.3f}, p={sh_p:.4f}  ->",
          "normalité OK" if sh_p > 0.05 else "normalité violée")

    bp = het_breuschpagan(resid, X)
    print(f"Breusch-Pagan (homoscédasticité)       : LM p-value={bp[1]:.4f}  ->",
          "homoscédasticité OK" if bp[1] > 0.05 else "homoscédasticité violée")

    print(f"Durbin-Watson (indépendance résidus)   : {sm.stats.stattools.durbin_watson(resid):.2f}  "
          "(proche de 2 = résidus indépendants)")

    # ==========================================================================
    # 6. TEST DE MODÉRATION — H4 (l'ancienneté d'usage modère l'effet de TAM sur PERF)
    # ==========================================================================
    print("\n=== 6. TEST DE MODÉRATION (H4 : ANC modère la relation TAM -> PERF) ===")
    df["TAM_c"] = df["TAM"] - df["TAM"].mean()          # centrage des variables avant interaction
    df["ANC_c"] = df["ANC"] - df["ANC"].mean()
    df["TAM_x_ANC"] = df["TAM_c"] * df["ANC_c"]

    mod = smf.ols("PERF ~ TAM_c + ANC_c + TAM_x_ANC", data=df).fit()
    print(mod.params.round(3))
    print("p-values:", mod.pvalues.round(4).to_dict())
    print("Interaction significative (p < 0.05) ->", mod.pvalues["TAM_x_ANC"] < 0.05)

    # ==========================================================================
    # 7. ANALYSE DE MÉDIATION EN CHAÎNE
    #    Teste la structure causale du modèle conceptuel : TOE -> TAM -> RBV -> PERF
    #    (méthode du bootstrap, 5000 réplications par défaut sous pingouin)
    # ==========================================================================
    print("\n=== 7. ANALYSE DE MÉDIATION EN CHAÎNE ===")

    print("\n-- Médiation 1 : TOE -> TAM -> Performance --")
    med1 = pg.mediation_analysis(data=df, x="TOE", m="TAM", y="PERF", seed=42)
    print(med1.round(4))

    print("\n-- Médiation 2 : TAM -> RBV -> Performance --")
    med2 = pg.mediation_analysis(data=df, x="TAM", m="RBV", y="PERF", seed=42)
    print(med2.round(4))

    # ==========================================================================
    # 8. SYNTHÈSE — DÉCISION SUR LES HYPOTHÈSES DE RECHERCHE (seuil 5 %)
    # ==========================================================================
    print("\n=== 8. SYNTHÈSE : DÉCISION SUR LES HYPOTHÈSES ===")

    def verdict(p_value, coef, seuil=0.05):
        if p_value < seuil and coef > 0:
            return "VALIDÉE (effet positif significatif)"
        elif p_value < seuil and coef < 0:
            return "EFFET INVERSE significatif (à discuter, cf. colinéarité)"
        else:
            return "NON CONFIRMÉE (non significatif à 5 %)"

    print("H1 (TAM -> Performance)        :", verdict(m2.pvalues["TAM"], m2.params["TAM"]))
    print("H2 (TOE -> Performance)        :", verdict(m3.pvalues["TOE"], m3.params["TOE"]))
    print("H3 (RBV -> Performance)        :", verdict(m3.pvalues["RBV"], m3.params["RBV"]))
    print("H4 (ANC modère TAM -> Performance):", verdict(mod.pvalues["TAM_x_ANC"], mod.params["TAM_x_ANC"]))

    # ==========================================================================
    # EXPORT DES DONNÉES COMPOSITES (pour analyses complémentaires / graphiques)
    # ==========================================================================
    df.to_csv(OUT_PATH, index=False)
    print(f"\n[OK] Fichier '{OUT_PATH}' écrit avec toutes les variables composites calculées.")


if __name__ == "__main__":
    try:
        main()
    except ModuleNotFoundError as e:
        print("=" * 78)
        print(f"ERREUR : un module Python nécessaire n'est pas installé ({e}).")
        print("Ouvrez une invite de commandes (cmd) et installez les dépendances avec :")
        print("    python -m pip install pandas numpy scipy statsmodels pingouin")
        print("=" * 78)
    except Exception:
        print("=" * 78)
        print("Une erreur est survenue pendant l'exécution du script :")
        print("=" * 78)
        traceback.print_exc()
    finally:
        input("\nAppuyez sur Entrée pour fermer cette fenêtre...")
