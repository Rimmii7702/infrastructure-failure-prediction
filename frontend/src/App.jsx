import { useState } from "react";
import axios from "axios";

const apiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim().replace(/\/+$/, "");

function App() {
  const [cpu, setCpu] = useState(50);
  const [memory, setMemory] = useState(50);
  const [disk, setDisk] = useState(50);
  const [temperature, setTemperature] = useState(50);
  const [network, setNetwork] = useState(500);
  const [request, setRequest] = useState({ status: "idle" });
  const { status, prediction, message } = request;
  const isLoading = status === "loading";

  const updateTelemetry = (setValue, value) => {
    if (isLoading) return;

    setValue(value);
    setRequest({ status: "idle" });
  };

  const predictFailure = async () => {
    if (isLoading) return;

    if (!apiBaseUrl) {
      setRequest({
        status: "error",
        message: "The API URL is missing. Set VITE_API_BASE_URL to the backend URL.",
      });
      return;
    }

    setRequest({ status: "loading" });
    try {
      const { data } = await axios.post(`${apiBaseUrl}/predict`, {
        cpu,
        memory,
        disk,
        temperature,
        network,
      });
      if (!Number.isFinite(data?.failure_probability)) {
        throw new Error("Invalid prediction response");
      }
      setRequest({ status: "success", prediction: data.failure_probability });
    } catch {
      setRequest({
        status: "error",
        message: "Could not get a prediction. Please try again.",
      });
    }
  };

  let riskLevel = "";

  if (status === "success") {
    if (prediction < 30) {
      riskLevel = "Healthy";
    } else if (prediction < 70) {
      riskLevel = "Warning";
    } else {
      riskLevel = "Critical";
    }
  }

  return (
    <div style={{ padding: "30px" }}>
      <h1>Predictive Server Failure System</h1>

      <h3>CPU: {cpu}%</h3>
      <input
        type="range"
        min="0"
        max="100"
        value={cpu}
        disabled={isLoading}
        onChange={(event) => updateTelemetry(setCpu, event.target.valueAsNumber)}
      />

      <h3>Memory: {memory}%</h3>
      <input
        type="range"
        min="0"
        max="100"
        value={memory}
        disabled={isLoading}
        onChange={(event) => updateTelemetry(setMemory, event.target.valueAsNumber)}
      />

      <h3>Disk: {disk}%</h3>
      <input
        type="range"
        min="0"
        max="100"
        value={disk}
        disabled={isLoading}
        onChange={(event) => updateTelemetry(setDisk, event.target.valueAsNumber)}
      />

      <h3>Temperature: {temperature}°C</h3>
      <input
        type="range"
        min="20"
        max="95"
        value={temperature}
        disabled={isLoading}
        onChange={(event) => updateTelemetry(setTemperature, event.target.valueAsNumber)}
      />

      <h3>Network: {network}</h3>
      <input
        type="range"
        min="0"
        max="1000"
        value={network}
        disabled={isLoading}
        onChange={(event) => updateTelemetry(setNetwork, event.target.valueAsNumber)}
      />

      <br /><br />

      <button type="button" onClick={predictFailure} disabled={isLoading}>
        Predict Failure
      </button>

      <div role="status">
        {status === "loading" && <p>Requesting a prediction…</p>}
        {status === "success" && (
          <>
            <h2>Failure Probability: {prediction}%</h2>
            <h2
              style={{
                color: riskLevel === "Critical" ? "red" : riskLevel === "Warning" ? "orange" : "green",
              }}
            >
              Status: {riskLevel}
            </h2>
          </>
        )}
      </div>
      {status === "error" && <p role="alert">{message}</p>}
    </div>
  );
}

export default App;
