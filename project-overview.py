import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import folium
    from folium.plugins import Fullscreen
    import pandas as pd
    import geopandas as gpd
    import altair as alt
    from shapely.geometry import Polygon
    import branca.colormap as cm
    import pysd
    import numpy as np
    return Fullscreen, alt, cm, folium, gpd, json, mo, np, pd, pysd


@app.cell
def _(gpd, np, pd, pysd):
    # Population and geo data
    wuerzburg_gdf = gpd.read_file("./Data/wuerzburg_bevoelkerung_100m.geojson")
    potential_locations_gdf = gpd.read_file("./Data/apl_candidates_clusters.geojson")


    # Simulation data
    sd_model = pysd.read_vensim("./Vensim-Model/APL-SFD-Würzburg-V3.mdl")
    model_overview = sd_model.doc
    simulation_results = sd_model.run().reset_index()

    constant_variables = ["Population", "Population growth rate", '"E-shopper share"', ''"E-shoppers growth rate"'', "Online purchase growth rate", "APL market growth rate", '"Avg. parcels per APL per month"']
    dynamic_variables = ["Market Size", '"Potential e-customers"', "APL users", "Online purchases per year", "Number of deliveries", "Number of APLs"]

    filtered_simulation_results = simulation_results[dynamic_variables].copy()

    for col in dynamic_variables:
        filtered_simulation_results[col] = filtered_simulation_results[col].apply(np.floor)


    # Demand Scenarios
    scenario_df = pd.read_csv("./Data/generated_scenarios.csv", sep=";")
    scenario_df["Year"] = scenario_df.index
    scenario_df = scenario_df.melt(id_vars="Year", var_name="Scenario", value_name="Demand")
    scenario_df['Scenario_Number'] = scenario_df['Scenario'].str.replace('Scenario', '').astype(int)

    # Optimization results
    results_df = pd.read_csv("./Data/combined_results_with_setup_costs.csv", sep=";")
    apl_ids = results_df['APL_ID'].drop_duplicates()
    cust_ids = results_df['Customer_ID'].drop_duplicates()
    apl_gdf = wuerzburg_gdf[wuerzburg_gdf['Gitter_ID_100m'].isin(apl_ids)].copy()
    cust_gdf = wuerzburg_gdf[wuerzburg_gdf['Gitter_ID_100m'].isin(cust_ids)].copy()
    apl_gdf = apl_gdf.rename(columns={'Gitter_ID_100m': 'APL_ID', 'geometry': 'APL_geometry'})
    cust_gdf = cust_gdf.rename(columns={'Gitter_ID_100m': 'Customer_ID', 'geometry': 'Customer_geometry'})
    results_df = results_df.merge(apl_gdf[['APL_ID', 'APL_geometry']], on='APL_ID', how='left')
    results_df = results_df.merge(cust_gdf[['Customer_ID', 'Customer_geometry']], on='Customer_ID', how='left')
    summary_df = pd.read_csv("./Data/cflp_apl_deployment_summary.csv", sep=";")
    return (
        filtered_simulation_results,
        potential_locations_gdf,
        results_df,
        scenario_df,
        simulation_results,
        summary_df,
        wuerzburg_gdf,
    )


@app.cell
def _(json):
    # Load DHL-Packstationen
    with open('./Data/dhl-parcel-lockers_wuerzburg.json', 'r', encoding='utf-8') as f:
        packstations = json.load(f)

    # Jetzt ist `packstations` wieder eine Liste von Dicts
    print(f"{len(packstations)} Packstationen geladen.")
    print(packstations[0])  # Beispielausgabe einer Station
    return (packstations,)


@app.cell
def _(gpd, mo):
    def heuristic_APLs(gdf, anzahl=1, mindestabstand_zellen=3):
        gdf_sorted = gdf.sort_values(by="Einwohner", ascending=False).copy()

        # Calculate grid-coordinates
        gdf_sorted['row'] = ((gdf_sorted['y_mp_100m'] - gdf_sorted['y_mp_100m'].min()) / 100).astype(int)
        gdf_sorted['col'] = ((gdf_sorted['x_mp_100m'] - gdf_sorted['x_mp_100m'].min()) / 100).astype(int)

        belegte_zellen = []
        ausgewaehlte = []

        for _, row in gdf_sorted.iterrows():
            r, c = row['row'], row['col']

            # Check distance to existing APLs
            too_close = any(
                abs(r - br) <= mindestabstand_zellen and abs(c - bc) <= mindestabstand_zellen
                for br, bc in belegte_zellen
            )

            if not too_close:
                belegte_zellen.append((r, c))
                ausgewaehlte.append(row)
                if len(ausgewaehlte) >= anzahl:
                    break


        return gpd.GeoDataFrame(ausgewaehlte, crs=gdf.crs)

    anzahl_slider = mo.ui.slider(start=0, stop=50, value=0, step=1, label="Number of APLs")
    return anzahl_slider, heuristic_APLs


@app.cell
def _(mo, packstations):
    mo.md(
        f"""
    #Würzburg Status Quo: Population and APL Landscape

    ##Population Map Würzburg

    Before starting the project, I created this interactive map to get an intuition for the problem. The map shows the population density of the city of Würzburg at the level of 100m x 100m grid cells according to the Zensus 2022 data. Additionally, you can activate two layers:

    * **DHL 'Packstationen':** The locations of existing DHL parcel lockers serve as a qualitative real-world benchmark (There are currently {(len(packstations))} DHL parcel lockers in Würzburg)
    * **Heuristic APL locations:** As a second, quantitative benchmark, I applied a heurisitc method based on population density to identify potentially optimal APL locations. This method places APLs in the most densely populated areas while maintaining a minimum distance between locations. You can setup APLs according to this heuristic with the interactive slider.

    When setting up approximately 40 APLs with the heuristic approach, the placement already looks pretty similar to the DHL setup. In the following, I will evaluate whether a more sophisticated optimization approach results in better reliability and cost-efficiency compared to the simple benchmark.
    """
    )
    return


@app.cell
def _(
    Fullscreen,
    anzahl_slider,
    cm,
    folium,
    heuristic_APLs,
    mo,
    packstations,
    wuerzburg_gdf,
):
    # Apply heuristic
    anzahl = anzahl_slider.value  # Vom Slider holen
    gewaehlte_packzellen = heuristic_APLs(wuerzburg_gdf, anzahl=anzahl)


    # Create map
    m = folium.Map(location=[49.7925, 9.9380], zoom_start=13, tiles='cartodbpositron')
    Fullscreen().add_to(m)


    # Population layer 100m
    colormap = cm.linear.YlOrRd_09.scale(0, 200)
    colormap.caption = 'Population per grid-cell'

    layer_100m = folium.FeatureGroup(name='Population 100m-Grid', overlay=True, control=True)
    folium.GeoJson(
        './Data/wuerzburg_bevoelkerung_100m.geojson',
        name='Population 100m-grid',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Einwohner']),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['Einwohner'], aliases=['Einwohner:']),
    ).add_to(layer_100m)
    layer_100m.add_to(m)


    # DHL Layer
    dhl_layer = folium.FeatureGroup(name='DHL Parcel Lockers', show=False)

    for i, ps in enumerate(packstations):
        folium.Marker(
        location=[ps['lat'], ps['lon']],
        popup=f"{ps['name']}<br>{ps['address']}",
        icon=folium.Icon(color='blue', icon='envelope', prefix='glyphicon')
        ).add_to(dhl_layer)

    dhl_layer.add_to(m)


    # Dynamic APL Layer (heuristisch)
    heuristic_layer = folium.FeatureGroup(name='Heuristic APL Locations', show=True)

    for _, row in gewaehlte_packzellen.iterrows():
        centroid = row['geometry'].centroid
        folium.Marker(
            location=[centroid.y, centroid.x],
            popup=f"Einwohner: {row['Einwohner']}",
            icon=folium.Icon(color='green', icon='envelope')
        ).add_to(heuristic_layer)

    heuristic_layer.add_to(m)

    folium.LayerControl(collapsed=True, position='topleft').add_to(m)

    mo.vstack([
        anzahl_slider,
        m
    ], gap=2, align='center')
    return colormap, dhl_layer


@app.cell
def _(mo):
    mo.vstack([
        mo.md(
            r"""
            #System Dynamics Model  
            In the next step, I adapted the Stocks and Flows Diagram by Rabe et al. to Würzburg to simulate the demand for APL deliveries over the next 10 years using Vensim.
            """),
        mo.image(src="./Images/APL-SFD-Würzburg-V3.png",
                width=450, style={"margin-right": "auto", "margin-left": "auto"}),
    ])
    return


@app.cell
def _(
    apl_users_chart,
    deliveries_chart,
    filtered_simulation_results,
    mo,
    purchases_chart,
):
    mo.vstack([
        mo.md("##Simulation results"),
        mo.hstack([
            # market_size_chart,
            apl_users_chart,
            purchases_chart,
            deliveries_chart,
        ]),
        mo.callout(filtered_simulation_results),
    ], align="center")
    return


@app.cell
def _(alt, simulation_results):
    deliveries_chart = alt.Chart(simulation_results).mark_line().encode(
        x=alt.X("time:Q"),
        y=alt.Y("Number of deliveries:Q")
    )

    apl_users_chart = alt.Chart(simulation_results).mark_line().encode(
        x=alt.X("time:Q"),
        y=alt.Y("APL users")
    )

    purchases_chart = alt.Chart(simulation_results).mark_line().encode(
        x=alt.X("time:Q"),
        y=alt.Y("Online purchases per year:Q")
    )

    return apl_users_chart, deliveries_chart, purchases_chart


@app.cell
def _(mo):
    mo.md(
        r"""
    ##Demand Scenarios

    After getting an idea of the demand development over the next 10 years, I created 20 demand scenarios following a normal distribution, with increasing uncertainty in future periods. Demand refers to the number of deliveries to APLs per year. The scenarios are ordered from pessimistic to optimistic. 
    The scenario- and time-dependent demand is modeled as a normally distributed random variable:

    $$
    D_{ts} \sim \mathcal{N}(\mu_{ts}, \sigma_{ts}^2)
    $$

    with

    $$
    \mu_{ts} = D_t \cdot f_s
    \quad \text{and} \quad
    \sigma_{ts} = \delta \cdot t \cdot \mu_{ts}
    $$

    where $\delta = 0.01$ is a scaling parameter for the uncertainty,  
    $D_t$ is the simulated base demand in month $t$, and  
    $f_s$ is a scenario-dependent factor linearly spaced between 0.9 and 1.1.
    """
    )
    return


@app.cell
def _(interactive_chart, mo, scenario_selection):
    mo.vstack([
        interactive_chart,
        scenario_selection
    ], gap=2, align="center")
    return


@app.cell
def _(alt, mo, scenario_df):
    # Scenario-selection ui-element

    color_scale = alt.Scale(scheme="redyellowgreen")
    scenario_options = scenario_df["Scenario"].unique()
    default_scenarios = ["Scenario1", "Scenario10", "Scenario20"]

    scenario_selection = mo.ui.multiselect(
        scenario_options, label="Select the scenarios you want to display", value=default_scenarios
    )
    return color_scale, scenario_selection


@app.cell
def _(alt, color_scale, scenario_df, scenario_selection):
    # Demand-scenario chart

    filtered_df = scenario_df[scenario_df["Scenario"].isin(scenario_selection.value)]

    interactive_chart = alt.Chart(filtered_df).mark_line().encode(
        x=alt.X("Year:Q", title="Year"),
        y=alt.Y("Demand:Q", title="Demand"),
        color=alt.Color("Scenario_Number:N", scale=color_scale, legend=alt.Legend(title="Scenario")),
        tooltip=['Scenario', 'Year', 'Demand'] 
    ).properties(
        width=700,
        height=400,
        title="Generated demand scenarios"
    )
    return (interactive_chart,)


@app.cell
def _(mo, summary_df):
    mo.md(
        f"""
    # Optimization

    In the next step, I modeled the problem as a Capacitated **Facility Location Problem (CFLP)**. The initial dataset consists of 100-meter grid cells across the city of Würzburg, each representing a potential facility site or demand point. However, including all grid cells as possible facility locations would result in an intractably large number of decision variables.  
    To address this issue, I used clustering techniques to reduce the candidate locations to **60 population-based clusters**, which serve as aggregated potential APL sites. For the demand sites, I started of with a **300-meter grid**. The resulting optimization model is implemented in Pyomo and solved using the CPLEX solver.  
    I solve the problem over a ten-year time horizon.
    In total, **{summary_df['Total_APLs_Opened'].sum()} APLs** are deployed for a defined base case.
    """
    )
    return


@app.cell
def _(
    Fullscreen,
    colormap,
    dhl_layer,
    folium,
    potential_locations_gdf,
    results_df,
    summary_df,
):
    # Create second map

    m1 = folium.Map(location=[49.7925, 9.9380], zoom_start=13, tiles='cartodbpositron')
    Fullscreen().add_to(m1)
    dhl_layer.add_to(m1)

    # Population layer 300m
    layer_300m = folium.FeatureGroup(name='Population 300m-Grid', overlay=True, control=True, show=True)
    folium.GeoJson(
        './Data/wuerzburg_bevoelkerung_300m.geojson',
        name='Population 300m-Grid',
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['Einwohner']),
            'color': 'black',
            'weight': 0.3,
            'fillOpacity': 0.7,
        },
        tooltip=folium.GeoJsonTooltip(fields=['Einwohner'], aliases=['Einwohner:']),
    ).add_to(layer_300m)

    layer_300m.add_to(m1)

    # Potential Locations Layer
    cluster_layer = folium.FeatureGroup(name='Potential APL Locations (Population Clusters)', show=True)
    apl_centroids = potential_locations_gdf.copy()
    apl_centroids["geometry"] = apl_centroids["geometry"].centroid

    for idx, pt in apl_centroids.iterrows():
        lat1, lon1 = pt['geometry'].y, pt['geometry'].x
        folium.CircleMarker(
            location=(lat1, lon1),
            popup=f"APL-ID:{pt['Gitter_ID_100m']}",
            radius=4,
            color="blue",
            fill=True,
            fill_opacity=0.8
        ).add_to(cluster_layer)

    cluster_layer.add_to(m1)

    # Deployed APLs Layer
    selected_apls = summary_df["APL_ID"].tolist()
    deployed_apls = apl_centroids[apl_centroids["Gitter_ID_100m"].isin(selected_apls)]

    selected_layer = folium.FeatureGroup(name="Deployed APLs", show=False)

    for _, apl in deployed_apls.iterrows():
        popup_text = f"APL {apl}"
        folium.CircleMarker(
            location=[apl['geometry'].y, apl['geometry'].x],
            radius=4,
            color="#7a92f2",
            fill=True,
            popup=popup_text
        ).add_to(selected_layer)

    selected_layer.add_to(m1)

    # Underperformer
    underutilized_apls = summary_df[summary_df["Underutilized_Most_Periods"] == True]["APL_ID"].tolist()
    highlighted_apls = apl_centroids[apl_centroids["Gitter_ID_100m"].isin(underutilized_apls)]

    low_utilization_layer = folium.FeatureGroup(name="Low capacity utilization APLs", show=False)

    for _, apl in highlighted_apls.iterrows():
        folium.Marker(
            location=[apl['geometry'].y, apl['geometry'].x],
            popup="Underutilized in >50% of periods",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(low_utilization_layer)

    low_utilization_layer.add_to(m1)

    # APL-Customer connections
    for t in sorted(results_df['Period'].unique()):
        group = folium.FeatureGroup(name=f"Period {t}", show=False)

        df_t = results_df[results_df['Period'] == t]

        for apl_id, sub_df in df_t.groupby("APL_ID"):
            apl_point = sub_df['APL_geometry'].iloc[0].centroid
            popup_text = f"APL {apl_id}<br>Kunden: {sub_df['Customer_ID'].nunique()}"
            folium.CircleMarker(
                location=[apl_point.y, apl_point.x],
                radius=4,
                color="#7a92f2",
                fill=True,
                fill_opacity=0.8,
                popup=popup_text
            ).add_to(group)

            # Verbindungen zu Kunden
            for _, row_ in sub_df.iterrows():
                apl_centroid = row_['APL_geometry'].centroid
                cust_centroid = row_['Customer_geometry'].centroid
                folium.PolyLine(
                    locations=[[apl_centroid.y, apl_centroid.x],
                               [cust_centroid.y, cust_centroid.x]],
                    color="#7a92f2",
                    weight=1
                ).add_to(group)

        group.add_to(m1)

    folium.LayerControl(collapsed=True, position="bottomleft").add_to(m1)

    m1
    return


@app.cell
def _(mo, pd):
    evaluation_df = pd.read_csv("./Data/robustness-analysis.csv", sep=";")
    relevant_columns = ['Test Case', 'Parameter', 'Number of opened APLs', 'Combined Reliability', 'Combined costs', 'Remark']
    evaluation_df = evaluation_df[relevant_columns].copy()

    mo.md(r"""
        # Results
        To evaluate different APL placement strategies, I conducted a comprehensive reliability and cost analysis across a variety of parameter settings and demand scenarios. I defined a base case and then systematically altered one parameter at a time in the subsequent test cases.

        For reliability evaluation I conducted a **Monte Carlo Simulation**:

        * For each of the 100 simulation runs, I generated random demand values for each grid cell and time period. 
        * These values were drawn from a normal distribution, using the SD model output as the expected value and a standard deviation of 20%, and distributed proportionally to the population density of each customer cell.

        If an APL’s capacity was exceeded, I recorded both the frequency of overloads and the amount of unsatisfied demand. From this, I derived a **Combined Reliability** score

        The **Combined Costs** consist of the setup costs, derived in the CFLP and the variable delivery costs. I calculated the delivery costs using real-life routes, derived from the Open Source Routing Machine (OSRM) API and a **Capacitated Vehicle Routing Problem (CVRP)**.

        """)

    return (evaluation_df,)


@app.cell
def _(alt, evaluation_df, mo):
    # Farbskala definieren
    color_scale_dots = alt.Scale(
        domain=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        range=['red', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange', 'orange','red']
    )

    # Selection for interactivity
    selection = alt.selection_point(name="selected", fields=["Test Case"])

    # Base chart for the circles
    base = alt.Chart(evaluation_df).mark_circle(size=100).encode(
        x=alt.X("Combined costs:Q", title="Combined Costs"),
        y=alt.Y("Combined Reliability:Q", title="Combined Reliability"),
        color=alt.condition(selection, "Test Case:N", alt.value("lightgray"), scale=color_scale_dots),
        tooltip=["Test Case:N", "Combined costs:Q", "Combined Reliability:Q"]
    ).add_params(selection)

    # Text labels
    text = alt.Chart(evaluation_df).mark_text(
        align='center',
        baseline='middle',
        fontSize=11,
        dx=-9,  # Nudge the text to the right
        dy=0   # Nudge the text up
    ).encode(
        x=alt.X("Combined costs:Q"),
        y=alt.Y("Combined Reliability:Q"),
        text="Test Case:N",
        color=alt.condition(selection, alt.value("black"), alt.value("black"))
    )

    # Combine the layers
    chart = (base + text).properties(
        width=600, height=400, title="Reliability vs Total Costs"
    )

    chart = mo.ui.altair_chart(chart)

    return (chart,)


@app.cell
def _(chart):
    selected = chart.selections
    return (selected,)


@app.cell
def _(evaluation_df, selected):
    selected_cases = selected.get("selected", {}).get("Test Case", [])
    print(selected_cases)
    filtered_evaluation_df = evaluation_df[evaluation_df["Test Case"].isin(selected_cases)]
    return (filtered_evaluation_df,)


@app.cell
def _(chart, filtered_evaluation_df, mo):
    mo.callout(
        mo.vstack([
            mo.md(
                r"""
                Click on data points in the chart for more information. Hold <kbd>Shift</kbd> and <kbd>Left Click</kbd> to select multiple points.
                """),
            filtered_evaluation_df,
            chart
        ])
    )

    return


if __name__ == "__main__":
    app.run()
