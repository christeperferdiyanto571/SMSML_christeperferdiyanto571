
import requests

# URL model (sesuaikan port jika berbeda)
URL = "http://localhost:5001/invocations"

def test():
    data = {
        "dataframe_split": {
            "columns": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude"],
            "data": [[8.3, 41.0, 6.98, 1.02, 2228.0, 2.55, 37.88, -122.24]]
        }
    }
    r = requests.post(URL, json=data)
    print(r.json())

if __name__ == "__main__":
    test()
