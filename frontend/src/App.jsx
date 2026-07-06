import { useState } from "react";
import axios from "axios";

function App() {

  const [cpu,setCpu] = useState(50);
  const [memory,setMemory] = useState(50);
  const [disk,setDisk] = useState(50);
  const [temperature,setTemperature] = useState(50);
  const [network,setNetwork] = useState(500);

  const [prediction,setPrediction] = useState(null);

  const predictFailure = async () => {

    const response = await axios.post(
  "https://infrastructure-failure-prediction-2.onrender.com/predict",
  {
    cpu: Number(cpu),
    memory: Number(memory),
    disk: Number(disk),
    temperature: Number(temperature),
    network: Number(network),
  }

    );

    setPrediction(
      response.data.failure_probability
    );
  };

  let riskLevel = "";

  if (prediction !== null) {
    if (prediction < 30) {
      riskLevel = "Healthy";
    } else if (prediction < 70) {
      riskLevel = "Warning";
    } else {
      riskLevel = "Critical";
    }
  }

  return (
    <div style={{padding:"30px"}}>

      <h1>Predictive Server Failure System</h1>

      <h3>CPU: {cpu}%</h3>
      <input
        type="range"
        min="0"
        max="100"
        value={cpu}
        onChange={(e)=>setCpu(Number(e.target.value))}
      />

      <h3>Memory: {memory}%</h3>
      <input
        type="range"
        min="0"
        max="100"
        value={memory}
        onChange={(e)=>setMemory(Number(e.target.value))}
      />

      <h3>Disk: {disk}%</h3>
      <input
        type="range"
        min="0"
        max="100"
        value={disk}
        onChange={(e)=>setDisk(Number(e.target.value))}
      />

      <h3>Temperature: {temperature}°C</h3>
      <input
        type="range"
        min="20"
        max="100"
        value={temperature}
        onChange={(e)=>setTemperature(Number(e.target.value))}
      />

      <h3>Network: {network}</h3>
      <input
        type="range"
        min="0"
        max="1000"
        value={network}
        onChange={(e)=>setNetwork(Number(e.target.value))}
      />

      <br /><br />

      <button onClick={predictFailure}>
        Predict Failure
      </button>

      {prediction !== null && (
  <>
    <h2>
      Failure Probability: {prediction}%
    </h2>

    <h2
  style={{
    color:
      riskLevel === "Critical"
        ? "red"
        : riskLevel === "Warning"
        ? "orange"
        : "green"
  }}
>
  Status: {riskLevel}
</h2>
  </>
)}

    </div>
  );
}

export default App;