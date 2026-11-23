import json
import unittest

import httpx
import pandas as pd

from backend.api import app, model


class PredictionAPITest(unittest.IsolatedAsyncioTestCase):
    telemetry = {"cpu": 95, "memory": 90, "disk": 70, "temperature": 88, "network": 500}
    ranges = {
        "cpu": (0, 100),
        "memory": (0, 100),
        "disk": (0, 100),
        "temperature": (20, 95),
        "network": (0, 1000),
    }

    async def asyncSetUp(self):
        self.client = await self.enterAsyncContext(
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://testserver")
        )

    async def test_health(self):
        response = await self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "running"})

    async def test_valid_predictions_match_saved_model(self):
        fractional = {"cpu": 25.5, "memory": 40.25, "disk": 0, "temperature": 37.5, "network": 1000}
        for telemetry in (self.telemetry, fractional):
            with self.subTest(telemetry=telemetry):
                # The model uses the feature order above regardless of JSON key order.
                probability = model.predict_proba(pd.DataFrame([telemetry]))[0][1]
                expected = {"failure_probability": round(float(probability * 100), 2)}
                payload = dict(reversed(telemetry.items()))

                response = await self.client.post("/predict", json=payload)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json(), expected)

    async def test_accepts_inclusive_range_boundaries(self):
        for index in (0, 1):
            payload = {field: bounds[index] for field, bounds in self.ranges.items()}
            with self.subTest(payload=payload):
                response = await self.client.post("/predict", json=payload)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(set(response.json()), {"failure_probability"})
                self.assertIsInstance(response.json()["failure_probability"], (int, float))

    async def test_rejects_values_outside_each_range(self):
        for field, (minimum, maximum) in self.ranges.items():
            for value, error in (
                (minimum - 0.01, "greater_than_equal"),
                (maximum + 0.01, "less_than_equal"),
            ):
                with self.subTest(field=field, value=value):
                    response = await self.client.post("/predict", json={**self.telemetry, field: value})

                    self.assertEqual(response.status_code, 422)
                    detail = response.json()["detail"][0]
                    self.assertEqual(detail["loc"], ["body", field])
                    self.assertEqual(detail["type"], error)

    async def test_missing_fields(self):
        for field in self.telemetry:
            with self.subTest(field=field):
                payload = self.telemetry.copy()
                del payload[field]

                response = await self.client.post("/predict", json=payload)

                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"][0]["loc"], ["body", field])

    async def test_rejects_non_numeric_values(self):
        for field in self.telemetry:
            for value in ("95", "high", True, None, [], {}):
                with self.subTest(field=field, value=value):
                    response = await self.client.post("/predict", json={**self.telemetry, field: value})

                    self.assertEqual(response.status_code, 422)
                    self.assertEqual(response.json()["detail"][0]["loc"], ["body", field])

    async def test_rejects_non_finite_values(self):
        payload = json.dumps({**self.telemetry, "cpu": "overflow"}).replace('"overflow"', "1e309")

        response = await self.client.post("/predict", content=payload, headers={"Content-Type": "application/json"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "cpu"])

    async def test_rejects_non_object_body(self):
        response = await self.client.post("/predict", json=[self.telemetry])

        self.assertEqual(response.status_code, 422)
