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
def _(demand_series, np, pd):
    # Szenario-Anzahl und Parameter
    num_scenarios = 20
    num_months = 120
    delta = 0.003

    # Faktoren zur Skalierung der Nachfrage je Szenario (linear von pessimistisch bis optimistisch)
    scenario_factors = np.linspace(0.9, 1.1, num_scenarios)  # z.B. 80% bis 120% des Basiswerts

    # 5. Matrix vorbereiten
    scenarios = np.zeros((num_months, num_scenarios))

    # 6. Szenarien generieren
    for s_idx, factor in enumerate(scenario_factors):
        for t in range(1, num_months + 1):  # t = 1...120
            mu = factor * demand_series[t - 1]  # Nachfrage angepasst durch Szenariofaktor
            sigma = delta / np.log(t + 1) * t * mu  # Unsicherheit steigt mit der Zeit
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
        num_months,
        num_scenarios,
        s_idx,
        scenario_df,
        scenario_factors,
        scenarios,
        sigma,
        t,
    )


@app.cell
def _(alt, delta, np, num_months, pd):
    # Erstelle eine Liste von t-Werten
    t_values = np.arange(1, num_months + 1)

    # Berechne die gedämpften Delta-Werte
    damped_delta_values = delta / np.log(t_values + 1)

    # Erstelle ein DataFrame
    data = pd.DataFrame({'t': t_values, 'damped_delta': damped_delta_values})

    # Erstelle den Altair-Plot
    chart = alt.Chart(data).mark_line().encode(
        x=alt.X('t', title='Zeit (Monate)'),
        y=alt.Y('damped_delta', title='Gedämpftes Delta')
    ).properties(
        title='Logarithmische Dämpfung von Delta'
    )

    chart
    return chart, damped_delta_values, data, t_values


if __name__ == "__main__":
    app.run()
