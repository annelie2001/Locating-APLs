import marimo

__generated_with = "0.13.15"
app = marimo.App(width="medium")


@app.cell
def _():
    from dotenv import load_dotenv
    import os
    import requests
    import json
    return json, load_dotenv, os, requests


@app.cell
def _(json, load_dotenv, os, requests):
    latitude = 49.7913
    longitude = 9.9534

    load_dotenv()
    dhl_api_key = os.getenv("SECRET_KEY_DHL")

    url = 'https://api.dhl.com/location-finder/v1/find-by-geo'
    headers = {
        'DHL-API-Key': dhl_api_key,
        'Accept': 'application/json'
    }

    # request parameters
    params_dhl = {
        'latitude': latitude,
        'longitude': longitude,
        'radius': 5000,  
        'limit': 50,
        'locationType': 'locker'
    }

    response_dhl = requests.get(url, headers=headers, params=params_dhl)

    packstations = []
    if response_dhl.status_code == 200:
        data_dhl = response_dhl.json()
        for location in data_dhl.get('locations', []):
            geo = location.get('place', {}).get('geo', {})
            address = location.get('place', {}).get('address', {})
            lat = geo.get('latitude')
            lon = geo.get('longitude')

            if lat and lon:
                packstations.append({
                    'name': location.get('name', 'Packstation'),
                    'lat': lat,
                    'lon': lon,
                    'address': f"{address.get('streetAddress', '')}, {address.get('postalCode', '')} {address.get('addressLocality', '')}"
                })
        with open('./Data/dhl-parcel-lockers_wuerzburg.json', 'w', encoding='utf-8') as f:
            json.dump(packstations, f, ensure_ascii=False, indent=2)

        print("Packstationen gespeichert.")

    else:
        print(f"Fehler beim Abrufen der Packstationen: {response_dhl.status_code} - {response_dhl.text}")
    return


if __name__ == "__main__":
    app.run()
