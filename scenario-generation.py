import marimo

__generated_with = "0.11.22"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import marimo as mo
    import altair as alt
    return alt, mo, np, pd


@app.cell
def _(pd):
    # Daten einlesen
    df = pd.read_csv("./Data/results-sim2.csv", sep=";")
    return (df,)


@app.cell
def _(df):
    demand_series = df["Number of deliveries : S2"].values  # Länge: 120 Monate
    print(demand_series[:12])
    return (demand_series,)


@app.cell
def _(demand_series, np, num_months, pd):
    # Szenario-Anzahl und Parameter
    num_scenarios = 20
    num_scenarios = 120
    delta = 0.002

    # Faktoren zur Skalierung der Nachfrage je Szenario (linear von pessimistisch bis optimistisch)
    scenario_factors = np.linspace(0.9, 1.1, num_scenarios)  # z.B. 80% bis 120% des Basiswerts

    # 5. Matrix vorbereiten
    scenarios = np.zeros((num_months, num_scenarios))

    # 6. Szenarien generieren
    for s_idx, factor in enumerate(scenario_factors):
        for t in range(1, num_months + 1):  # t = 1...120
            mu = factor * demand_series[t - 1]  # Nachfrage angepasst durch Szenariofaktor
            sigma = (np.sqrt(3) / 3) * delta * t * mu  # Unsicherheit steigt mit der Zeit
            scenarios[t - 1, s_idx] = int(np.random.normal(loc=mu, scale=sigma))

    # 7. DataFrame mit Szenarien (jeder Reihe = ein Szenario)
    scenario_df = pd.DataFrame(scenarios,
                               columns=[f"Scenario{i+1}" for i in range(num_scenarios)])

    # 8. Speichern
    scenario_df.to_csv("./Data/generated_scenarios.csv", sep=";", decimal='.', index=False)
    print("Szenarien generiert und in 'generated_scenarios.csv' gespeichert.")
    return (
        delta,
        factor,
        mu,
        num_scenarios,
        s_idx,
        scenario_df,
        scenario_factors,
        scenarios,
        sigma,
        t,
    )


if __name__ == "__main__":
    app.run()
