import requests

BASE_URL = "https://api.db.nomics.world/v22"

class DBnomicsClient:
    def __init__(self, timeout=30):
        self.timeout = timeout

    def _get(self, endpoint, params=None):
        url = f"{BASE_URL}{endpoint}"
        r = requests.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---- METADATA ----
    def list_datasets(self, provider_code):
        return self._get(f"/providers/{provider_code}")

    def list_series(self, provider_code, dataset_code, limit=1000, offset=0):
        return self._get(
            f"/series/{provider_code}/{dataset_code}",
            params={"limit": limit, "offset": offset}
        )

    # ---- DATA ----
    def get_series(self, provider_code, dataset_code, series_code):
        return self._get(
            f"/series/{provider_code}/{dataset_code}/{series_code}"
        )

# def clean_providers_query(d):
    