

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import numpy as np
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
    return (pd,)


@app.cell
def _():
    # ----------------------------
    # Parameter
    # ----------------------------
    vehicle_capacity = 4000
    num_vehicles = 5

    hub_coords = (9.999129, 49.772268) # noch durch gesnappte Koordinaten austauschen, bisher nicht gespeichert
    return


@app.cell
def _(pd):
    # === Daten einlesen ===
    df_dist = pd.read_csv("./Data/cvrp_distance_matrix.csv", index_col=0, sep=";")

    # Spaltennamen bereinigen
    df_dist.columns = df_dist.columns.str.replace(r"^to_apl_", "", regex=True)
    df_dist.index = df_dist.index.str.replace(r"^to_apl_", "", regex=True)

    # Bedarfe laden und nur aktive APLs behalten
    df_apl = pd.read_csv("./Data/cflp_apl_deployment_summary.csv", sep=";")
    active_apls = df_apl["APL_ID"].tolist()
    distance_matrix = df_dist.loc[active_apls, active_apls]
    print(distance_matrix.shape)
    return


if __name__ == "__main__":
    app.run()
