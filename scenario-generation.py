

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import marimo as mo
    import altair as alt
    import pysd
    return np, pd, pysd


@app.cell
def _(pd):
    # Daten einlesen
    df = pd.read_csv("./Data/results-model2-sim1.csv", sep=";")
    return


@app.cell
def _(pysd):
    sd_model = pysd.read_vensim("./Vensim-Model/APL-SFD-Würzburg-V3.mdl")
    simulation_results = sd_model.run().reset_index()
    simulation_results
    return (simulation_results,)


@app.cell
def _(simulation_results):
    demand_series = simulation_results["Number of deliveries"].values
    return (demand_series,)


@app.cell
def _(demand_series, np, pd):
    # Szenario-Anzahl und Parameter
    num_scenarios = 20
    num_periods = 10
    delta = 0.01

    # Faktoren zur Skalierung der Nachfrage je Szenario (linear von pessimistisch bis optimistisch)
    scenario_factors = np.linspace(0.9, 1.1, num_scenarios)  # z.B. 80% bis 120% des Basiswerts

    # 5. Matrix vorbereiten
    scenarios = np.zeros((num_periods, num_scenarios))

    # 6. Szenarien generieren
    for s_idx, factor in enumerate(scenario_factors):
        for t in range(1, num_periods + 1):
            mu = factor * demand_series[t - 1]  # Nachfrage angepasst durch Szenariofaktor
            # sigma = delta / np.log(t + 1) * t * mu   # Dämpfung
            sigma = delta * t * mu  # Unsicherheit steigt mit der Zeit
            scenarios[t - 1, s_idx] = int(np.random.normal(loc=mu, scale=sigma))

    # 7. DataFrame mit Szenarien (jeder Reihe = ein Szenario)
    scenario_df = pd.DataFrame(scenarios,
                               columns=[f"Scenario{i+1}" for i in range(num_scenarios)])

    # 8. Speichern
    scenario_df.to_csv("./Data/generated_scenarios.csv", sep=";", decimal='.', index=False)
    print("Szenarien generiert und in 'generated_scenarios.csv' gespeichert.")
    return


if __name__ == "__main__":
    app.run()
